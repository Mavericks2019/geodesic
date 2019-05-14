[T,Y]=ode45('geodesic',[0,3],[pi/3,1.1152,-2*pi/3,-0.8164]);
plot(T,Y(:,1));
plot(T,Y(:,2));
plot(T,Y(:,3))
plot(T,Y(:,4))
plot(Y(:,1),Y(:,3))
plot(cos(Y(:,1))+cos(Y(:,1)+Y(:,3)),sin(Y(:,1))+sin(Y(:,1)+Y(:,3)))