function aoa = computeAoAFromPosition(pos_tag,pos_anchor)
  n_anchor = size(pos_anchor,2);
  n_smp = size(pos_tag,2);  
  aoa = zeros(n_anchor,n_smp);
  for i = 1:n_anchor
    aoa(i,:) = atan2(pos_anchor(2,i) - pos_tag(2,:),pos_anchor(1,i) - pos_tag(1,:));
  end
end