#require "torch.toplevel";;
#require "core";;
open Torch
open Core

(* This is only for undirected models. It also removes duplicates*)
let get_map_from_undirected_edges edges = 
	let l2 = List.map edges ~f:(fun (x,y) -> (y,x)) in
	let l = edges @ l2 in 
	let m = Map.Poly.of_alist_fold l ~init:[] ~f:(fun l n -> n::l) in
	Map.Poly.map m List.stable_dedup

(* And this one is only for chain graphs *)
let get_chain_graph_map_from_size n_words =
    get_map_from_undirected_edges (List.zip_exn 
  			          (List.range 0 ((List.length n_words) - 1))
  				  (List.range 1 (List.length n_words)))

(* Convenience function for linear chain MRFs, to be used in tandem with
get_chain_graph_map_from_size: with indices generated from 
get_chain_graph_map_from_size, generate a lookup table for pairwise potentials
(0,1)-> 0, (1,0) -> 1, (1,2) -> 2, (2,1) -> 3
(2,3) -> 4, (3,2) -> 5, (3,4) -> 6, ...
*)
let linear_chain_MRF_tuple_to_idx (x,y) = 
    if y > x then 2 * x else 2 * x - 1

(* Just a function where all pairwise potentials are the same *)
let get_pairwise_potential_function poly_map n_words potentials =
    let lookup idx1 idx2 = potentials in
    lookup

(* For a map such as from get_map_from_undirected_edges, get all tuples of edges (a,b) where a in the map, and b in map[a] *)
let get_message_keys map = List.map (Map.Poly.keys map) ~f:(fun k -> List.map (Map.Poly.find_exn map k) ~f:(fun v -> (k,v))) |> List.concat;;
let get_message_keys_single map idx = List.map ~f:(fun v -> (idx, v)) (Map.Poly.find_exn map idx);;

let get_messages_to message_map target_idx = Map.Poly.filter_keys message_map ~f:(fun (_, x2) -> (x2 = target_idx));;
let get_messages_to_except message_map target_idx except_idx = Map.Poly.filter_keys message_map ~f:(fun (x1, x2) -> (x2 = target_idx && x1 <> except_idx));;
let get_product_of_messages_to message_map target_idx = List.reduce_exn ~f:Tensor.mul (Map.Poly.data (get_messages_to message_map target_idx));;
let get_product_of_messages_to_except_exn message_map target_idx except_idx = List.reduce_exn ~f:Tensor.mul (Map.Poly.data (get_messages_to_except message_map target_idx except_idx));;
let get_product_of_messages_to_except message_map target_idx except_idx = List.reduce ~f:Tensor.mul (Map.Poly.data (get_messages_to_except message_map target_idx except_idx));;

let add_some_noise ?(noise=0.00000001) x = Tensor.add x (Tensor.normal_ ~mean:0. ~std:noise (Tensor.ones_like x));;
let normalize x = Tensor.div x  (Tensor.sum x)

class mutable_MRF words_ (priors_: Tensor.t) (pairwise_potential_: Tensor.t) verbose_ = object(self)
  val words = words_
  val priors = priors_
  val pairwise_potential = pairwise_potential_
  val n_words = List.length words_
  val verbose = verbose_
  val neighbour_map = get_map_from_undirected_edges (List.zip_exn 
  						    (List.range 0 ((List.length words_) - 1))
  						    (List.range 1 (List.length words_)))
  method get_neighbour_map = neighbour_map  						    

  method get_neighbours idx = match (idx >= 0 && idx < n_words) with
  	true -> Map.Poly.find_exn neighbour_map idx  
	| _    -> []

  method get_univariate_potential_array idx = match (idx >= 0 && idx < n_words) with 
  	true ->  let x = Tensor.get_float1 priors idx in 
                 Tensor.float_vec [1. -. x; x]
        | false -> Tensor.float_vec [0.; 0.]

  method get_pairwise_potential_array (idx1 : int) (idx2 : int) = pairwise_potential

  method get_initial_messages =
	Map.Poly.of_alist_exn (List.map (get_message_keys neighbour_map) ~f:(fun k -> (k, Tensor.float_vec [0.5; 0.5;])))

  method get_initial_belief idx =
	let univariate = self#get_univariate_potential_array idx in
	let message_keys = get_message_keys_single neighbour_map idx in
	let all_initial_messages = self#get_initial_messages in
	let messages = List.map message_keys ~f:(fun x-> Map.Poly.find_exn all_initial_messages x) in
	List.fold ~init:univariate ~f:Tensor.mul messages

  method get_belief messages idx =
  	let univariate = self#get_univariate_potential_array idx in
  	let message_keys = get_message_keys_single neighbour_map idx in
  	let message_subset = List.map message_keys ~f:(fun x-> Map.Poly.find_exn messages x) in
	List.fold ~init:univariate ~f:Tensor.mul message_subset

  method get_message messages (source_idx : int) (target_idx : int)  =
	let m = Tensor.mm (self#get_univariate_potential_array target_idx) pairwise_potential in
	let m = match (get_product_of_messages_to_except messages source_idx target_idx) with
	Some x -> Tensor.mul m x
	| None -> m in
	normalize m

  method make_inference ?(noise=0.00000001) ?(eps=0.000001) =
	let noisy_map_dist l1 l2 = List.map2_exn ~f:Tensor.dist
						 (l1 |> Map.Poly.data)
						 (l2 |> Map.Poly.data |> List.map ~f:(fun x -> add_some_noise ~noise:noise x)) |>
			      List.reduce_exn ~f:Tensor.add |> Tensor.float_value in
  	let initial_messages = self#get_initial_messages in
  	let init_dist = Float.max_finite_value in 
  	let rec iter messages cur_dist iter_number =
	    let new_messages = Map.Poly.mapi messages 
	                                     ~f:(fun ~key:(source_idx, target_idx) ~data:_ ->
	                                      self#get_message messages source_idx target_idx) in 
	    let new_dist = noisy_map_dist messages new_messages in
	    let diff = (Float.abs (Float.sub cur_dist new_dist)) in
  	    if (Float.compare diff eps) > 0 then iter new_messages new_dist (iter_number + 1) else (messages, iter_number) in
  	iter initial_messages init_dist 0

  method make_inference_and_get_beliefs =
  	let messages, _ = self#make_inference ~noise:0.00000001 ~eps:0.000001 in
  	self#get_neighbour_map |> Map.Poly.keys |> List.map ~f:(fun node_idx -> self#get_belief messages node_idx |> normalize)
end
;;
