function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.

G_sigma = 0;

for cfg = 1:size(visible_state, 2)
  visible_cfg = visible_state(:,cfg);
  for hidden_node = 1:size(rbm_w, 1);
    weight_vector = rbm_w(hidden_node,:);
    G_sigma += weight_vector * visible_cfg * hidden_state(hidden_node,cfg);
  end
end

G = G_sigma/size(visible_state, 2);

end
