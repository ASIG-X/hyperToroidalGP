function plotErrorDist(idx_te,lb_SE,ub_SE,err_te,err_gauss,lw)
  figure('Position',[0,0,420,420]);
  hold on
  inBetween = [lb_SE,fliplr(ub_SE)];
  t2 = [idx_te,fliplr(idx_te)];
  fill(t2,inBetween, 'k','FaceAlpha',0.2,'EdgeColor','none');  
  scatter(idx_te,err_te,20,'filled','MarkerFaceColor','r','MarkerFaceAlpha',1)
  plot(idx_te,err_gauss,'Color','b','LineWidth',lw);
  ax = gca; 
  ax.FontSize = 20;   
  set(gca,'TickLabelInterpreter','latex')
  grid on
  box on
  set(gca,'GridLineStyle','--')
  xlabel('time step')
  ylabel('error [m]')
  ylim([-0.5,1])
  % xlim([0,180]);
  % pbaspect([1 1 1])
  % set(gca,'xtick',0:50:180);
  % print(f,'-dpdf',str);
end