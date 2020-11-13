let mrf1 = new mutable_MRF ["my"; "name"; "is"; "nikolaos"] (Tensor.float_vec [0.5; 0.3; 0.4; 0.9]) (Tensor.stack [(Tensor.float_vec [0.6; 0.4;]); (Tensor.float_vec [0.4; 0.6;])] ~dim:0) false;;
mrf1#get_univariate_potential_array 0;;
mrf1#get_initial_messages;;
mrf1#get_neighbours 0;;
mrf1#get_initial_messages;;
get_message_keys_single (mrf1#get_neighbour_map) 1;;
mrf1#get_initial_belief 0;;
get_product_of_messages_to_except mrf1#get_initial_messages 2 0;;
get_product_of_messages_to_except mrf1#get_initial_messages 2 1;;
get_product_of_messages_to mrf1#get_initial_messages 0;;
get_product_of_messages_to mrf1#get_initial_messages 1;;
mrf1#get_message mrf1#get_initial_messages 0 1;;
mrf1#get_message mrf1#get_initial_messages 1 0;;
let beliefs = mrf1#make_inference_and_get_beliefs;;

let mrf1 = new mutable_MRF ["my"; "name"; "is"; "nikolaos"] (Tensor.float_vec [0.5; 0.3; 0.4; 0.9]) (Tensor.stack [(Tensor.float_vec [0.6; 0.4;]); (Tensor.float_vec [0.4; 0.6;])] ~dim:0) false;;
let beliefs = mrf1#make_inference_and_get_beliefs;;
  	
let new_messages = Map.Poly.mapi messages ~f:(fun ~key:(source_idx, target_idx) ~data:_-> mrf1#get_message messages source_idx target_idx) ;;
