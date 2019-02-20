clc;clear all;close all;
%program to generate qpsk signal
b=input('Enter the bit stream');
ps=1;
w=2*pi*60;
t=0:.126:2*pi;
l=length(b);

    for i=1:1:l
        if (b(1,i)==0)
        bt(1,i)=-1;
        else if (b(1,i)==1)
                bt(i)=1;
                else
                    error('input bit stream is non binary');
                    break;
            end
    end

    end

%  now we have to form the odd and the even bit stream
s=1;
for i=1:2:l
    odd(s)=bt(i);
    s=s+1;
end
dd=1;
for j=2:2:l
    even(1,dd)=bt(1,j);
    dd=dd+1;
end

k=0;
lo=length(odd);

le=length(even);
ls=max(lo,le);
if (le < ls)
    even=[even,-1];
end
if (lo < ls)
    odd=[odd,-1];
end
lo=length(odd);

le=length(even);
cc=0;
for i=1:1:lo
    if (odd(i)==1)
        cc=cc+1;
        no(cc)=1;
        cc=cc+1;
        no(cc)=1;
       
    
    else if (odd(i)==-1)
            cc=cc+1;
            no(cc)=-1;
            cc=cc+1;
        no(cc)=-1;
        end
    end
end

cc=0;
for i=1:1:le
    if (even(i)==1)
        cc=cc+1;
        ne(cc)=1;
        cc=cc+1;
        ne(cc)=1;
    
    else if (even(i)==-1)
            cc=cc+1;
        ne(cc)=-1;
        cc=cc+1;
        ne(cc)=-1;
        end
    end
end
s1=sqrt(ps)*sin(t);
s0=sqrt(ps)*cos(t);
  k=0;  
nl=ls*50;
        for j=1:1:ls
            for i=1:1:50
                k=k+1;
                v(i)= no(j)*s1(i)+ne(j)*s0(i);
                vm(k)=v(i);
                
            end
        end
n=1;
tb=0;
for m=1:1:l
    for i=1:1:25
        
        tb(n)=bt(m);
        odstr(n)= no(m);
        
        evstr(n)= ne(m);
        n=n+1;
    end
end

n=1;
for m=1:1:ls
    for i=1:1:50
        odstr(n)= no(m);
        evstr(n)= ne(m);
        n=n+1;
    end
end

              
        subplot(4,1,1);stairs(tb ,'k','LineWidth',2);
        title('INPUT BIT STREAM');
%           grid on; 
        subplot(4,1,2);stairs(odstr,'r','LineWidth',2);
        title('ODD BIT STREAM');
%           grid on; 
        subplot(4,1,3);stairs(evstr,'g','LineWidth',2);
        title('EVEN BIT STREAM');
%           grid on; 
        subplot(4,1,4);plot(vm,'b','LineWidth',2);
        title('QPSK WAVEFORM')

%program for qpsk demodulation
aa=input('Enter the quadrant of the qpsk signal(1 for first, 2 for second, 3 for third and 4 for fourth');

ll=length(aa);
for i=1:1:ll
    if (aa(1,i)==1)
        be(1,i)=(-1);
        bo(1,i)=1;
    else if (aa(1,i)==2)
          be(1,i)=1;
          bo(1,i)=1;
        else if (aa(1,i)==3)
                 be(1,i)=1;
                 bo(1,i)=(-1);
            else if (aa(1,i)==4)
                    be(1,i)=(-1);
                    bo(1,i)=(-1);
                else
                    error('Quadrant can only have values 1,2,3 and 4')
                    break;
                end
            end
        end
    end
end
 ss=length(be);
 i=1;j=1;
while (i<ss)
    d(j)=bo(1,i);
    j=j+1;
    d(j)=be(1,i);
    i=i+1;
    j=j+1;
end
l=length(d);
for k=1:1:l
    if d(1,k)==-1
        d(1,k)=0;
    end
end
figure
subplot(3,1,1);stem(bo);title('Odd bit stream');
subplot(3,1,2);stem(be);title('Even bit stream');
subplot(3,1,3);stem(d);title('Original bit stream');
