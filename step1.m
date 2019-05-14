clear
syms x y p q g1 g2 a b g
%运动学方程
x=cos(p)+cos(p+q);
y=sin(p)+sin(p+q);
%运动学方程%
a=diff(x,p);
b=diff(y,p);
c=diff(x,q);
d=diff(y,q);
%mactrix%
g1=[a;b];G1=[a b];
g2=[c;d];G2=[c d];
g=[G1;G2]*[g1 g2];
G=inv(g);
sg=simplify(g);
sG=simplify(G);
%mactrix%
K=[p q];
r=0;
for i=1:2;
    for j=1:2 ;
        for k=1:2;
            for t=1:2;
            a=1/2*sG(i,t)*(diff(sg(t,j),K(1,k),1)+diff(sg(k,t),K(1,j),1)-diff(sg(j,k),K(1,t),1));
            r=r+a;
            end
            R(i,j,k)= r;
            r=0;
        end
    end
end
S=simplify(R);