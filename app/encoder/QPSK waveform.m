%QPSK waveform for binary sequence 011010

no_of_cycles = 7;
time=0:0.01:no_of_cycles;          
fc=1;

%Generating the sinusoidal signal of frequency 1 hz
X=sin(2*pi*fc*time);

%Generating the cosinusoidal signal of frequency 1 hz
Y=cos(2*pi*fc*time);

%Generating the odd sequence bo

bo=1.*[time>=0];
bo=bo-2.*[time<2];
bo=bo-2.*[time>6];


%Generating the even sequence be

be=1.*[time>=0];
be=be-2.*[time<1];
be=be-2.*[time>3];


%Modulate sinusoidal and cosinusoidal signals with even and odd sequences respectively 
%To generate the BPSK signals for even and odd parts of Input binary
%sequence

bpsk_odd=bo.*Y;
bpsk_even=be.*X;
figure('Name','Odd sequence and waveform'),subplot(3,1,1),plot(time,bo,'r'),axis([0 7 -2 2]),grid on,title('Odd sequence');
subplot(3,1,2),plot(time,Y,'r'),grid on,title('Cosinusoidal waveform');
subplot(3,1,3),plot(time,bpsk_odd,'r'),
grid on,title('BPSK Waveform for odd sequence');
figure('Name','Even sequence and waveform'),subplot(3,1,1),plot(time,be,'r'),axis([0 7 -2 2]),grid on,title('Even sequence');
subplot(3,1,2),plot(time,X,'r'),grid on,title('Sinusoidal waveform');
subplot(3,1,3),plot(time,bpsk_even,'b'),
grid on,title('BPSK Waveform for even sequence');


%Add two BPSK signals to generate QPSK signal 
sum=bpsk_odd+bpsk_even;
figure('Name','BPSK and QPSK waveforms');subplot(3,1,1),plot(time,bpsk_odd,'r'),
grid on,title('BPSK Waveform for odd sequence');
subplot(3,1,2),plot(time,bpsk_even,'b'),
grid on,title('BPSK Waveform for even sequence');
subplot(3,1,3),plot(time,sum,'r'),title('QPSK Waveform');
grid on;
