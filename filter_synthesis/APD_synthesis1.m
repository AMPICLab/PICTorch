%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code synthesizes All-Pass Decomposition (APD) based filters
% These map to a ring based optical filters with two allpass branches in an 
% outer MZI
%
% Ref: C. K. Madsen, "Efficient Architectures for Exactly realizing 
% Optical Filters with Optimum Bandpass Designs," IEEE Photonics Technology
% Letters, vol. 10, no. 8, Aug 1998
%
% (c) Vishal Saxena 2015
% Last modified: June/21/2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

% Generate Chebyshev II IIR Filter Polynomial
order = 4;
Rs = 40; %  dB of stopband attenuation down from the peak passband value 
%Wn = [3e9 4e9]/20e9;% stopband edge frequency
Wn = 0.13;% stopband edge frequency
[z,p,k] = cheby2(order, Rs, Wn, 'low');
%[z,p,k] = butter(order,Wn,'low');
% sos = zp2sos(z,p,k);
% fvtool(sos,'Analysis','freq')

[b,a] = zp2tf(z,p,k)
w = linspace(-pi, 3*pi, 2^12);
h = freqz(b,a,w);
%[h,w] = freqz(b,a,2^14);
G = filt(b,a)

%zplane(b,a)

figure()
plot(w*20/pi,mag2db(abs(h)), 'r', 'lineWidth', 2)
axis([-20 20 -70 5])
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




