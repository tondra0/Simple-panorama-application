function [m1, m2] = select_matches(d1, d2, num)
    d1 = zscore(d1')';
    d2 = zscore(d2')';
    
    % get distance
    [ndata, dimx] = size(d1);
    [ncentres, dimc] = size(d2);
    if dimx ~= dimc
    	error('Data dimension does not match dimension of centres')
    end
    
    distance = (ones(ncentres, 1) * sum((d1.^2)', 1))' + ones(ndata, 1) * sum((d2.^2)',1) - 2.*(d1*(d2'));
    
    % Rounding errors occasionally cause negative entries
    if any(any(distance<0))
      distance(distance<0) = 0;
    end
        
    [h,w] = size(distance);
    distance = reshape(distance,1,[]);
    [~,idx] = sort(distance);
    [r,c] = ind2sub([h,w],idx(1:num));
    m1 = r';
    m2 = c';
    
end