clear
syms p q X Y R r right1 right2
%S,T是 p q 的导数%
z=[p X q Y];
r1=[ sin(q)/(cos(q) - 1); -1/sin(q)];
r2=[ -1/sin(q);-1/sin(q)];
R1=[ -(2*sin(q))/(cos(q) - 1); -sin(q)/(cos(q) - 1)];
R2=[ -sin(q)/(cos(q) - 1); -sin(q)/(cos(q) - 1)];
rt=[r1 r2];
RT=[R1 R2];
for i=1:2
    for j=1:2
        right1=rt(i,j)*z(2*i)*z(2*j)+right1;
        right2=RT(i,j)*z(2*i)*z(2*j)+right2;
    end
end
right1
right2