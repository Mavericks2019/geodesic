function dz=geodesic(s,z)
dz=zeros(4,1);
dz(1)=z(2);
dz(2)=-(sin(z(3))*z(2)^2)/(cos(z(3)) - 1) + (2*z(2)*z(4))/sin(z(3)) + z(4)^2/sin(z(3));
dz(3)=z(4);
dz(4)=(2*z(2)^2*sin(z(3)))/(cos(z(3)) - 1) + (z(4)^2*sin(z(3)))/(cos(z(3)) - 1) + (2*z(2)*z(4)*sin(z(3)))/(cos(z(3)) - 1);