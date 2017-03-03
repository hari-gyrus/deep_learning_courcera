function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

visible_state = sample_bernoulli(visible_data);

% positive phase 1st time with clamped visible data

hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state);
hidden_state = sample_bernoulli(hidden_probability);
d_G_by_rbm_w_pos1 = configuration_goodness_gradient(visible_state, hidden_state);

% negative phase with hidden_state from positive phase

visible_probability = hidden_state_to_visible_probabilities(rbm_w, hidden_state);
visible_state = sample_bernoulli(visible_probability);

% positive phase 2nd time with clamped visible data

hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state);

code_for_question_7 = 0

if code_for_question_7
  hidden_state = sample_bernoulli(hidden_probability);
  d_G_by_rbm_w_pos2 = configuration_goodness_gradient(visible_state, hidden_state);
else
  d_G_by_rbm_w_pos2 = configuration_goodness_gradient(visible_state, hidden_probability);
  
% calculate gradient approximation produced by CD-1

ret = d_G_by_rbm_w_pos1 - d_G_by_rbm_w_pos2;

end
