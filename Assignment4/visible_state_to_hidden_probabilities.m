function hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% The returned value is a matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This takes in the (binary) states of the visible units, and returns the activation probabilities of the hidden units conditional on those states.
% error('not yet implemented');

for cfg = 1:size(visible_state, 2)
  visible_units = visible_state(:,cfg);
  for hidden_node = 1:size(rbm_w, 1);
    weight_vector = rbm_w(hidden_node,:);
    hidden_probability(hidden_node, cfg) = 1/(1 + exp(- weight_vector * visible_units));
  end
end

end
