function visible_probability = hidden_state_to_visible_probabilities(rbm_w, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% The returned value is a matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% This takes in the (binary) states of the hidden units, and returns the activation probabilities of the visible units, conditional on those states.

 for cfg = 1:size(hidden_state, 2)
  hidden_units = hidden_state(:,cfg);
  for visible_node = 1:size(rbm_w, 2);
    weight_vector = rbm_w(:,visible_node)';
    visible_probability(visible_node, cfg) = 1/(1 + exp(- weight_vector * hidden_units));
  end
end

end
