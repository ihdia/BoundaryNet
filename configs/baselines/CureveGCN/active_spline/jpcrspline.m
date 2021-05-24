function [xs,ys] = jpcrspline(ix, iy, n, al)

                                                % This function generates Catmull-Rom spline with
                                                % customizable parametization
                                                % The input vector x and y must be a row vector
                                                % al - parametization
                                                % (n - 1) points to be interpolated between two knots

                                                % Expand the input vector
x       = zeros(1,size(ix, 2)+2);
y       = zeros(1,size(iy, 2)+2);

x(1, 2:(size(ix,2)+1)) = ix;
y(1, 2:(size(ix,2)+1)) = iy;

x(1,1)  = ix(1,1)-0.1*(ix(1,2)-ix(1,1));
y(1,1)  = iy(1,1)-0.1*(iy(1,2)-iy(1,1));
x(1,end)= ix(1,end)+0.1*(ix(1,end)-ix(1,end-1));
y(1,end)= iy(1,end)+0.1*(iy(1,end)-iy(1,end-1));
              
np      = size(x, 2);                           % number of control points
ns      = np - 2;                               % the actual number of segments is ns - 1
t       = zeros(1, size(x, 2));

for i = 2:size(t, 2)                        	
    t(1, i) = (((x(1,i)-x(1,i-1))^2+(y(1,i)-y(1,i-1))^2)^(0.5))^al+t(1,i-1);
end
    lp      = 1;                                % loop counter
    xs      = zeros(1, n*(ns-1));
    ys      = zeros(1, n*(ns-1));

for sg  = 2:ns
    v       = t(1,sg):((t(1,sg+1)-t(1,sg))/n):t(1,sg+1);
    t0      = t(1,sg-1);
    t1      = t(1,sg);
    t2      = t(1,sg+1);
    t3      = t(1,sg+2);
    
    if isequal(sg, ns)
        n   = n+1;
    end
    
    for i   = 1:n
        tv  = v(1,i);
        
        x01 = (t1-tv)/(t1-t0)*x(1,sg-1)+(tv-t0)/(t1-t0)*x(1,sg);
        y01 = (t1-tv)/(t1-t0)*y(1,sg-1)+(tv-t0)/(t1-t0)*y(1,sg);
        
        x12 = (t2-tv)/(t2-t1)*x(1,sg)+(tv-t1)/(t2-t1)*x(1,sg+1);
        y12 = (t2-tv)/(t2-t1)*y(1,sg)+(tv-t1)/(t2-t1)*y(1,sg+1);
        
        x23 = (t3-tv)/(t3-t2)*x(1,sg+1)+(tv-t2)/(t3-t2)*x(1,sg+2);
        y23 = (t3-tv)/(t3-t2)*y(1,sg+1)+(tv-t2)/(t3-t2)*y(1,sg+2);
        
        x012= (t2-tv)/(t2-t0)*x01+(tv-t0)/(t2-t0)*x12;
        y012= (t2-tv)/(t2-t0)*y01+(tv-t0)/(t2-t0)*y12;
        
        x123= (t3-tv)/(t3-t1)*x12+(tv-t1)/(t3-t1)*x23;
        y123= (t3-tv)/(t3-t1)*y12+(tv-t1)/(t3-t1)*y23;
        
        xs(1,lp) = (t2-tv)/(t2-t1)*x012+(tv-t1)/(t2-t1)*x123;
        ys(1,lp) = (t2-tv)/(t2-t1)*y012+(tv-t1)/(t2-t1)*y123;
        
        
        lp  = lp + 1;
    end

    
end