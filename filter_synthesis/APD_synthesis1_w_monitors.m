%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code synthesizes All-Pass Decomposition (APD) based filters
% These map to a ring based optical filters with two allpass branches in an 
% outer MZI
%
% Ref: C. K. Madsen, "Efficient Architectures for Exactly realizing 
% Optical Filters with Optimum Bandpass Designs," IEEE Photonics Technology
% Letters, vol. 10, no. 8, Aug 1998
%
% Modifications for Dual-Bus Rings with Monitor
%
% (c) Vishal Saxena 2015
% Last modified: June/21/2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

% Generate Chebyshev II IIR Filter Polynomial
order = 2;
Rs = 40; %  dB of stopband attenuation down from the peak passband value 
%Wn = [3e9 4e9]/20e9;% stopband edge frequency
Wn = 0.13;% stopband edge frequency
%[z,p,k] = cheby2(order, Rs, Wn, 'low');
[z,p,k] = butter(order,Wn,'low');
% sos = zp2sos(z,p,k);
% fvtool(sos,'Analysis','freq')

[b,a] = zp2tf(z,p,k)
w = linspace(-pi, 3*pi, 2^12);
h = freqz(b,a,w);
%[h,w] = freqz(b,a,2^14);
G = filt(b,a)

%zplane(b,a)

figure()
plot(w*20/pi,mag2db(abs(h)),'lineWidth', 2)
%axis([-20 20 -70 5])
grid
title('Optical Filter Response');
xlabel('frequency (GHz)')
ylabel('Attenuation (dB)')

% Allpass Decomposition
% http://www.mathworks.com/help/dsp/ref/tf2ca.html

[d1,d2,ejbeta] = tf2ca(b,a) 	% TF2CA returns denominators of the allpass
beta = angle(ejbeta);

% Verify if the all-pass decomposition went Ok
num = 0.5*conj(ejbeta)*conv(fliplr(d1),d2)+0.5*ejbeta*conv(fliplr(d2),d1);
den = conv(d1,d2); 	% Reconstruct numerator and denonimator.
MaxDiff=max([max(b-num),max(a-den)]) % Compare original and reconstructed
                                     % numerator and denominators.

% Numerators are reversed conjugated polynomials of the denominators
n1 = fliplr(conj(d1));
n2 = fliplr(conj(d2));

% Find the all-pass functions
A1 = ejbeta*filt(n1, d1, 1)
A2 = conj(ejbeta)*filt(n2, d2, 1)

%% Map to photonic ring response
% Assuming losless single-bus rings

r = 0.998; % Ring loss factor

% Top Arm rings
zn1 = roots(d1)';
ord1 = size(zn1,2);
k1 = (1.-abs(zn1).^2)  % Coupling coefficients
t1 = sqrt(1.-k1);      % Coupler thru transmission
phi1 = angle(zn1);    % Phase shifts in the rings
phi_tot = sum(phi1)   % Total phase shift
km = 0.02;

% Bottom Arm rings
zn2 = roots(d2)';
ord2 = size(zn2,2);
k2 = (1.-abs(zn2).^2)   % Coupling coefficients
t2 = sqrt(1.-k2);      % Coupler thru transmission
phi2 = angle(zn2);     % Phase shifts in the rings


%% Verify the Ring Filter Response and Plot it

% Top Arm rings
Htop = filt([1], [1], 1);

figure();
hold on; grid;

for j = 1:ord1
    ejphi = exp(i*phi1(j));
    t = t1(j);
    sr = sqrt(k1(j));
    sm = sqrt(km);
    cr = sqrt(1-k1(j));
    cm = sqrt(1-km);
    %ejphi = exp(i*phi);

    bb = [0 -sr*sm*r*ejphi];
    aa = [1 -cr*cm*r*ejphi];
    Hr =  filt(bb, aa, 1);

    bb1 = [cr -cm*r*ejphi];
    Hm = filt(bb1, aa, 1);
   
    Htop = Htop*Hm;
    
    [nn, dd] = tfdata(Hr,'v');
    [nn2, dd2] = tfdata(Hm,'v');
    ww = linspace(-pi, pi, 2^12);
    hr = freqz(nn, dd, ww);
    hm = freqz(nn2, dd2, ww);

   % plot(ww*20/pi,mag2db(abs(hr)), 'b', 'lineWidth', 1)
    plot(ww*20/pi,mag2db(abs(hm)), 'lineWidth', 1.5)
end    

title('Optical Monitor Responses');
xlabel('frequency (GHz)')
ylabel('Attenuation (dB)')

% Bottom Arm rings

Hbot = filt([1], [1], 1);

for j = 1:ord2
    ejphi = exp(i*phi2(j));
    t = t2(j);
    sr = sqrt(k2(j));
    sm = sqrt(km);
    cr = sqrt(1-k2(j));
    cm = sqrt(1-km);
    %ejphi = exp(i*phi);

    bb = [0 -sr*sm*r*ejphi];
    aa = [1 -cr*cm*r*ejphi];
    Hr =  filt(bb, aa, 1);

    bb1 = [cr -cm*r*ejphi];
    Hm = filt(bb1, aa, 1);
    
    [nn, dd] = tfdata(Hr,'v');
    [nn2, dd2] = tfdata(Hm,'v');
    ww = linspace(-pi, pi, 2^12);
    hr = freqz(nn, dd, ww);
    hm = freqz(nn2, dd2, ww);

   % plot(ww*20/pi,mag2db(abs(hr)), 'b', 'lineWidth', 1)
    plot(ww*20/pi,mag2db(abs(hm)), 'lineWidth', 1.5)
    Hbot = Hbot*Hm;
end

% Plot Result Summary


fprintf('Top Arm Rings:\n');
fprintf('----------------------------\n');
disp('Coupler Thru transmission (t)');
disp(t1);
disp('Coupling coefficient (k)');
disp(k1);
disp('Phase shift (phi_n)');
disp(phi1);

fprintf('Bottom Arm Rings:\n');
fprintf('----------------------------\n');
disp('Coupler Thru transmission (t)');
disp(t2);
disp('Coupling coefficient (k)');
disp(k2);
disp('Phase shift (phi_n)');
disp(phi2);

fprintf('----------------------------\n');
disp('Filter beta');
disp(beta);
disp('Filter phi_tot');
disp(phi_tot);
disp('Filter beta - phi_tot');
disp(beta-phi_tot);


%% Complete Filter with couplers and phase shifter
% Tap at the bar output (Top-Bot)
GG = 0.5*(Htop*exp(i*(beta-phi_tot)) + Hbot*exp(i*(-beta+phi_tot)));
%GG = 0.5*(Htop*exp(i*(beta)) + Hbot*exp(i*(-beta)));
%GG = 0.5*j*(Htop + Hbot);
%GG = Htop;
[nn4, dd4] = tfdata(GG,'v');
[hh,ww] = freqz(nn4, dd4,2^14);

% figure()
% plot(ww/pi,mag2db(abs(hh)), 'r', 'lineWidth', 2)
% %plot(ww/pi,angle(hh), 'r', 'lineWidth', 2)
% %axis([0 1 -50 5])
% grid
% title('Cascade Ring Filter');
% xlabel('Normalized Frequency (\omega / \pi)')
% ylabel('Attenuation (dB)')



figure()

w = linspace(-pi, pi, 2^12);

[b, a] = tfdata(G,'v');
h1 = freqz(b,a,w);
figure()
zplane(b,a)

[b, a] = tfdata(GG,'v');
h2 = freqz(b,a,w);
figure()
zplane(b,a)

plot(w*20/pi,mag2db(abs(h1).^1), 'b-.', 'lineWidth', 2); hold on;
plot(w*20/pi,mag2db(abs(h2).^1), 'r', 'lineWidth', 2)
axis([-20 20 -70 5])
grid
% set(gca,'FontSize',12);
title('Optical Filter Response');
xlabel('frequency (GHz)')
ylabel('Attenuation (dB)')
legend('Ideal', 'Si-photonic');

