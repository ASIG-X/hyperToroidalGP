function plotTraj(pos_anchor,pos_tag_te,pos_est_vM)
  figure;
  scatter(pos_anchor(1,:),pos_anchor(2,:),80,'filled','MarkerFaceColor','#0a481e','MarkerFaceAlpha',0.8)
  hold on 
  plot(pos_tag_te(1,:),pos_tag_te(2,:),'r','LineWidth',2);  
  plot(pos_est_vM(1,:),pos_est_vM(2,:),'Color','b','LineWidth',2);
  xlabel('x [m]');
  ylabel('y [m]');
  ax = gca; 
  ax.FontSize = 22;   
  set(gca,'TickLabelInterpreter','latex')
  grid on
  box on
  set(gca,'GridLineStyle','--')
  axis equal  
  xlim([0,30])
  ylim([0,30])  
end