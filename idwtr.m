function  data = idwtr(wtr, L, filterh)
% function data = idwt(wtr, L, filterh); Calculates the IDWT of wavelet
% transformation wtr using wavelet filter  "filterh"  and  L  scales.  
% Use
%>>  t=linspace(0,1,1024); data = sqrt(t.*(1-t)).*sin((2*pi*1.05) ./(t+.05));
%>> filter=[sqrt(2)/2 sqrt(2)/2];
%>> max(abs(data - idwtr(dwtr(data,3,filter), 3,filter)))
% Inputs L and filterh need to match their counterparts from dwtr that
% produced input wtr
%ans = 3.8858e-016
nn = length(wtr);   n = length(filterh);         % Lengths
%if nargin < 3, L = round(log2(nn)); end;      % Depth of transformation
  H = fliplr(conj(filterh));                                       % Wavelet H filter
  G = conj(filterh); G(2:2:n) = -G(2:2:n);               % Wavelet G filter

LL = nn/(2^L);                                      % Number of scaling coeffs
C =  wtr(1:LL);                                     % Scaling coeffs
for j = 1:L                                             % Cascade algorithm
   w  = mod(0:n/2-1,LL)+1;                    % Make periodic
   D  = wtr(LL+1:2*LL);                          % Wavelet coeffs
   Cu(1:2:2*LL+n) = [C C(1,w)];             % Upsample & keep periodic
   Du(1:2:2*LL+n) = [D D(1,w)];             % Upsample & keep periodic
   C  = conv(Cu,H) + conv(Du,G);          % Convolve & add
   %C  = filter(H,[1],Cu) + filter(G,[1],Du); 
   C  = C([n:n+2*LL-1]-1);                    % Periodic part
   LL = 2*LL;                                       % Double the size of level
end;
data = C;                                             % The inverse DWT

