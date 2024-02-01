%% 已知量定义
clear all;
close all;
ACC_scale=1.5258789063E-06; %加速度计比例因子
GYO_scale=1.0850694444E-07; %陀螺仪比例因子
Samp_rate=100;              %采样率/Hz
g=9.7936174;                %重力值/m/(s^2)
omiga_e= 7.292115*1e-5;     %地球自转角速度/rad/s
Lat=deg2rad(30.531651244);  %已知维度/rad
%% 文件读取、数据预处理
Alig=readfile("duizhun.ASC");
Zshun=readfile("Zshun.ASC");
Zni=readfile("Z_g_f.ASC");
Yshun=readfile("Yshun.ASC");
Yni=readfile("Yni.ASC");
Xshun=readfile("Xshun.ASC");
Xni=readfile("Xni.ASC");
AzUp=readfile("AzUp.ASC");
AzDown=readfile("AzDown.ASC");
AxUp=readfile("AxUp.ASC");
AxDown=readfile("AxDown.ASC");
AyUp=readfile("AyUp.ASC");
AyDown=readfile("AyDown.ASC");
Zni_240=readfile("Zni.ASC");
% Xshun1=readfile("x_zheng.ASC");
% Xni1=readfile("x_fan.ASC");

Alig(:,1:3)=Alig(:,1:3)*ACC_scale*Samp_rate;
Alig(:,4:6)=Alig(:,4:6)*GYO_scale*Samp_rate;

Zshun(:,1:3)=Zshun(:,1:3)*ACC_scale*Samp_rate;
Zshun(:,4:6)=Zshun(:,4:6)*GYO_scale*Samp_rate;
Yshun(:,1:3)=Yshun(:,1:3)*ACC_scale*Samp_rate;
Yshun(:,4:6)=Yshun(:,4:6)*GYO_scale*Samp_rate;
Xshun(:,1:3)=Xshun(:,1:3)*ACC_scale*Samp_rate;
Xshun(:,4:6)=Xshun(:,4:6)*GYO_scale*Samp_rate;
% Xshun1(:,1:3)=Xshun1(:,1:3)*ACC_scale*Samp_rate;
% Xshun1(:,4:6)=Xshun1(:,4:6)*GYO_scale*Samp_rate;
Zni(:,1:3)=Zni(:,1:3)*ACC_scale*Samp_rate;
Zni(:,4:6)=Zni(:,4:6)*GYO_scale*Samp_rate;
Zni_240(:,1:3)=Zni_240(:,1:3)*ACC_scale*Samp_rate;
Zni_240(:,4:6)=Zni_240(:,4:6)*GYO_scale*Samp_rate;
Yni(:,1:3)=Yni(:,1:3)*ACC_scale*Samp_rate;
Yni(:,4:6)=Yni(:,4:6)*GYO_scale*Samp_rate;
Xni(:,1:3)=Xni(:,1:3)*ACC_scale*Samp_rate;
Xni(:,4:6)=Xni(:,4:6)*GYO_scale*Samp_rate;
% Xni1(:,1:3)=Xni1(:,1:3)*ACC_scale*Samp_rate;
% Xni1(:,4:6)=Xni1(:,4:6)*GYO_scale*Samp_rate;

AzUp(:,1:3)=AzUp(:,1:3)*ACC_scale*Samp_rate;
AzUp(:,4:6)=AzUp(:,4:6)*GYO_scale*Samp_rate;
AyUp(:,1:3)=AyUp(:,1:3)*ACC_scale*Samp_rate;
AyUp(:,4:6)=AyUp(:,4:6)*GYO_scale*Samp_rate;
AxUp(:,1:3)=AxUp(:,1:3)*ACC_scale*Samp_rate;
AxUp(:,4:6)=AxUp(:,4:6)*GYO_scale*Samp_rate;
AzDown(:,1:3)=AzDown(:,1:3)*ACC_scale*Samp_rate;
AzDown(:,4:6)=AzDown(:,4:6)*GYO_scale*Samp_rate;
AyDown(:,1:3)=AyDown(:,1:3)*ACC_scale*Samp_rate;
AyDown(:,4:6)=AyDown(:,4:6)*GYO_scale*Samp_rate;
AxDown(:,1:3)=AxDown(:,1:3)*ACC_scale*Samp_rate;
AxDown(:,4:6)=AxDown(:,4:6)*GYO_scale*Samp_rate;
%% 原始数据绘图检查
figure(1)
subplot(2,3,1);hold on;
plot(AxUp(:,3),'DisplayName','AxUp','LineWidth',1);plot(AxDown(:,3),'DisplayName','AxDown','LineWidth',1,'LineStyle','--');
title("加速度计x轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(2,3,2);hold on;
plot(-AyUp(:,2),'DisplayName','AyUp','LineWidth',1);plot(-AyDown(:,2),'DisplayName','AyDown','LineWidth',1,'LineStyle','--');
title("加速度计y轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(2,3,3);hold on;
plot(AzUp(:,1),'DisplayName','AzUp','LineWidth',1);plot(AzDown(:,1),'DisplayName','AzDown','LineWidth',1,'LineStyle','--');
title("加速度计z轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(2,3,4);hold on;
plot(Xshun(:,6),'DisplayName','Xshun','LineWidth',1);plot(Xni(:,6),'DisplayName','Xni','LineWidth',1,'LineStyle','--');
title("陀螺仪x轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(2,3,5);hold on;
plot(-Yshun(:,5),'DisplayName','Yshun','LineWidth',1);plot(-Yni(:,5),'DisplayName','Yni','LineWidth',1,'LineStyle','--');
title("陀螺仪y轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(2,3,6);hold on;
plot(Zshun(:,4),'DisplayName','Zshun','LineWidth',1);plot(Zni(:,4),'DisplayName','Zni','LineWidth',1,'LineStyle','--');
title("陀螺仪z轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;

figure(2)
subplot(2,3,1);plot(Alig(:,3),'DisplayName','f_x');
title("加速度计x轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend
subplot(2,3,2);plot(-Alig(:,2),'DisplayName','f_y');
title("加速度计y轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend
subplot(2,3,3);plot(Alig(:,1),'DisplayName','f_z');
title("加速度计z轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend
subplot(2,3,4);plot(Alig(:,6),'DisplayName','\omega_x');
title("陀螺仪x轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend
subplot(2,3,5);plot(-Alig(:,5),'DisplayName','\omega_x');
title("陀螺仪y轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend
subplot(2,3,6);plot(Alig(:,4),'DisplayName','\omega_x');
title("陀螺仪z轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend

%% 加速度计标定
f1=[g;0;0];
f2=[-g;0;0];
f3=[0;g;0];
f4=[0;-g;0];
f5=[0;0;g];
f6=[0;0;-g];
A=[f1,f2,f3,f4,f5,f6;1,1,1,1,1,1];%已知真值
M=[];
m0=[1,0,0,0;0,1,0,0;0,0,1,0];
meanxu=[mean(AxUp(:,3));-mean(AxUp(:,2));mean(AxUp(:,1))];
meanxd=[mean(AxDown(:,3));-mean(AxDown(:,2));mean(AxDown(:,1))];
meanyu=[mean(AyUp(:,3));-mean(AyUp(:,2));mean(AyUp(:,1))];
meanyd=[mean(AyDown(:,3));-mean(AyDown(:,2));mean(AyDown(:,1))];
meanzu=[mean(AzUp(:,3));-mean(AzUp(:,2));mean(AzUp(:,1))];
meanzd=[mean(AzDown(:,3));-mean(AzDown(:,2));mean(AzDown(:,1))];
L=[meanxu,meanxd,meanyu,meanyd,meanzu,meanzd];
m=L*A.'*inv((A*A.'));
%% 逐历元求，检查数据收敛性
%求平均值
meanaxu=zeros(3,1);
meanaxd=zeros(3,1);
meanayu=zeros(3,1);
meanayd=zeros(3,1);
meanazu=zeros(3,1);
meanazd=zeros(3,1);
for i=1:1:27000
    L0=[AxUp(i,3);-AxUp(i,2);AxUp(i,1)];
    meanaxu=(meanaxu*(i-1)+L0)/i;
    L0=[AxDown(i,3);-AxDown(i,2);AxDown(i,1)];
    meanaxd=(meanaxd*(i-1)+L0)/i;
    L0=[AyUp(i,3);-AyUp(i,2);AyUp(i,1)];
    meanayu=(meanayu*(i-1)+L0)/i;
    L0=[AyDown(i,3);-AyDown(i,2);AyDown(i,1)];
    meanayd=(meanayd*(i-1)+L0)/i;
    L0=[AzUp(i,3);-AzUp(i,2);AzUp(i,1)];
    meanazu=(meanazu*(i-1)+L0)/i;
    L0=[AzDown(i,3);-AzDown(i,2);AzDown(i,1)];
    meanazd=(meanazd*(i-1)+L0)/i;
    L=[meanaxu,meanaxd,meanayu,meanayd,meanazu,meanazd];

    %不求平均值，直接计算
%     f1_=[AxUp(i,3);-AxUp(i,2);AxUp(i,1)];
%     f2_=[AxDown(i,3);-AxDown(i,2);AxDown(i,1)];
%     f3_=[AyUp(i,3);-AyUp(i,2);AyUp(i,1)];
%     f4_=[AyDown(i,3);-AyDown(i,2);AyDown(i,1)];
%     f5_=[AzUp(i,3);-AzUp(i,2);AzUp(i,1)];
%     f6_=[AzDown(i,3);-AzDown(i,2);AzDown(i,1)];
%     L=[f1_,f2_,f3_,f4_,f5_,f6_];

    m0=L*A.'*inv((A*A.')); %#ok<*MINV> 
    %存储数据绘图
    M(i,1)=m0(1,1)-1;
    M(i,2)=m0(2,2)-1;
    M(i,3)=m0(3,3)-1;
    M(i,4)=m0(1,4);
    M(i,5)=m0(2,4);
    M(i,6)=m0(3,4);
    M(i,7)=m0(1,2);
    M(i,8)=m0(2,1);
    M(i,9)=m0(1,3);
    M(i,10)=m0(3,1);
    M(i,11)=m0(2,3);
    M(i,12)=m0(3,2);
end
%% 补偿IMU数据
m_inv=inv(m0(1:3,1:3));
b=m0(1:3,4);
for i=1:height(AxUp)
    m_c=m_inv*([AxUp(i,3);-AxUp(i,2);AxUp(i,1)]-b);
    xUp(i,1:3)=m_c.';
end
for i=1:height(AxDown)
    m_c=m_inv*([AxDown(i,3);-AxDown(i,2);AxDown(i,1)]-b);
    xDown(i,1:3)=m_c.';
end
for i=1:height(AyUp)
    m_c=m_inv*([AyUp(i,3);-AyUp(i,2);AyUp(i,1)]-b);
    yUp(i,1:3)=m_c.';
end
for i=1:height(AyDown)
    m_c=m_inv*([AyDown(i,3);-AyDown(i,2);AyDown(i,1)]-b);
    yDown(i,1:3)=m_c.';
end
for i=1:height(AzUp)
    m_c=m_inv*([AzUp(i,3);-AzUp(i,2);AzUp(i,1)]-b);
    zUp(i,1:3)=m_c.';
end
for i=1:height(AzDown)
    m_c=m_inv*([AzDown(i,3);-AzDown(i,2);AzDown(i,1)]-b);
    zDown(i,1:3)=m_c.';
end
%% 绘图
figure(3)
subplot(3,1,1);plot(M(:,1))
title("x轴加速度计比例因子误差");
xlabel("运转时间t/10^{-2}s")
ylabel("s_x")
subplot(3,1,2);plot(M(:,2))
title("y轴加速度计比例因子误差")
xlabel("运转时间t/10^{-2}s")
ylabel("s_y")
subplot(3,1,3);plot(M(:,3))
title("z轴加速度计比例因子误差")
xlabel("运转时间t/10^{-2}s")
ylabel("s_z")

figure(4)
subplot(3,1,1);plot(M(:,4)) 
title("x轴加速度计零偏");
xlabel("运转时间t/10^{-2}s")
ylabel("b_x")
subplot(3,1,2);plot(M(:,5))
title("y轴加速度计零偏");
xlabel("运转时间t/10^{-2}s")
ylabel("b_y")
subplot(3,1,3);plot(M(:,6))
title("z轴加速度计零偏");
xlabel("运转时间t/10^{-2}s")
ylabel("b_z")

figure(5)
subplot(2,3,1);plot(M(:,7))
title("交轴耦合误差");
xlabel("运转时间t/10^{-2}s")
ylabel("\gamma_y_x")
subplot(2,3,4);plot(M(:,8))
title("交轴耦合误差");
xlabel("运转时间t/10^{-2}s")
ylabel("\gamma_x_y")
subplot(2,3,2);plot(M(:,9))
title("交轴耦合误差");
xlabel("运转时间t/10^{-2}s")
ylabel("\gamma_z_x")
subplot(2,3,5);plot(M(:,10))
title("交轴耦合误差");
xlabel("运转时间t/10^{-2}s")
ylabel("\gamma_x_z")
subplot(2,3,3);plot(M(:,11))
title("交轴耦合误差");
xlabel("运转时间t/10^{-2}s")
ylabel("\gamma_z_y")
subplot(2,3,6);plot(M(:,12))
title("交轴耦合误差");
xlabel("运转时间t/10^{-2}s")
ylabel("\gamma_y_z")
%% IMU数据补偿绘图
figure(6)
subplot(3,3,1);hold on;
plot(AxUp(:,3),'DisplayName','AxUp','LineWidth',1);
title("补偿前加速度计x轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,2);hold on;
plot(-AyUp(:,2),'DisplayName','AyUp','LineWidth',1);
title("补偿前加速度计y轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,3);hold on;
plot(AzUp(:,1),'DisplayName','AzUp','LineWidth',1);
title("补偿前加速度计z轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,4);hold on;
plot(xUp(:,1),'DisplayName','AxUp','LineWidth',1);
title("补偿后加速度计x轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,5);hold on;
plot(yUp(:,2),'DisplayName','AyUp','LineWidth',1);
title("补偿后加速度计y轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,6);hold on;
plot(zUp(:,3),'DisplayName','AzUp','LineWidth',1);
title("补偿后加速度计z轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,7);hold on;
plot(AxUp(:,3)-xUp(:,1),'DisplayName','\deltaAxUp','LineWidth',1);
title("补偿前后加速度计x轴测量值差");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,8);hold on;
plot(-AyUp(:,2)-yUp(:,2),'DisplayName','\deltaAyUp','LineWidth',1);
title("补偿前后加速度计y轴测量值差");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,9);hold on;
plot(AzUp(:,1)-zUp(:,3),'DisplayName','\deltaAzUp','LineWidth',1);
title("补偿前后加速度计z轴测量值差");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;

figure(7)
subplot(3,3,1);hold on;
plot(AxDown(:,3),'DisplayName','AxDown','LineWidth',1,'LineStyle','--');
title("补偿前加速度计x轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,2);hold on;
plot(-AyDown(:,2),'DisplayName','AyDown','LineWidth',1,'LineStyle','--');
title("补偿前加速度计y轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,3);hold on;
plot(AzDown(:,1),'DisplayName','AzDown','LineWidth',1,'LineStyle','--');
title("补偿前加速度计z轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,4);hold on;
plot(xDown(:,1),'DisplayName','AxDown','LineWidth',1,'LineStyle','--');
title("补偿后加速度计x轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,5);hold on;
plot(yDown(:,2),'DisplayName','AyDown','LineWidth',1,'LineStyle','--');
title("补偿后加速度计y轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,6);hold on;
plot(zDown(:,3),'DisplayName','AzDown','LineWidth',1,'LineStyle','--');
title("补偿后加速度计z轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,7);hold on;
plot(AxDown(:,3)-xDown(:,1),'DisplayName','\deltaAxDown','LineWidth',1);
title("补偿前后加速度计x轴测量值差");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,8);hold on;
plot(-AyDown(:,2)-yDown(:,2),'DisplayName','\deltaAyDown','LineWidth',1);
title("补偿前后加速度计y轴测量值差");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,9);hold on;
plot(AzDown(:,1)-zDown(:,3),'DisplayName','\deltaAzDown','LineWidth',1);
title("补偿前后加速度计z轴测量值差");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
%% 陀螺标定
%取平稳的180°计算
[b_gx_180,delt_s_gx_180]=GYO(Xshun(:,6),Xni(:,6),1000,2800,1000,2800,180,18);
[b_gy_180,delt_s_gy_180]=GYO(-Yshun(:,5),-Yni(:,5),2000,3800,2000,3800,180,18);
[b_gz_180,delt_s_gz_180]=GYO(Zshun(:,4),Zni(:,4),1000,2800,2000,3800,180,18);

%使用别的小组数据进行全过程360°的计算
[b_gx_360,delt_s_gx_360]=GYO(Xshun(:,6),Xni(:,6),323,4523,303,4503,360,42);
[b_gy_360,delt_s_gy_360]=GYO(-Yshun(:,5),-Yni(:,5),85,4285,1065,5265,360,42);
[b_gz_360,delt_s_gz_360]=GYO(Zshun(:,4),Zni(:,4),212,4412,759,4959,360,42);

%s使用静态数据进行计算
[b_sg_x,delt_sg_x]=GYO1(AxUp(:,6),AxDown(:,6));
[b_sg_y,delt_sg_y]=GYO1(-AyUp(:,5),-AyDown(:,5));
[b_sg_z,delt_sg_z]=GYO1(AzUp(:,4),AzDown(:,4));
Xshun_c=((Xshun(:,6))-b_gx_360)/(1+delt_s_gx_360);
Xni_c=((Xni(:,6))-b_gx_360)/(1+delt_s_gx_360);
Yshun_c=((-Yshun(:,5))-b_gy_360)/(1+delt_s_gy_360);
Yni_c=((-Yni(:,5))-b_gy_360)/(1+delt_s_gy_360);
Zshun_c=((Zshun(:,4))-b_gz_360)/(1+delt_s_gz_360);
Zni_c=((Zni(:,4))-b_gz_360)/(1+delt_s_gz_360);
%% 
figure(8)
subplot(3,3,1);hold on;
plot(Xshun(:,6),'DisplayName','Xshun','LineWidth',1);
title("补偿前陀螺仪x轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,2);hold on;
plot(-Yshun(:,5),'DisplayName','Yshun','LineWidth',1);
title("补偿前陀螺仪y轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,3);hold on;
plot(Zshun(:,4),'DisplayName','Zshun','LineWidth',1);
title("补偿前陀螺仪z轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,4);hold on;
plot(Xshun_c,'DisplayName','Xshun','LineWidth',1);
title("补偿后陀螺仪x轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,5);hold on;
plot(Yshun_c,'DisplayName','Yshun','LineWidth',1);
title("补偿后陀螺仪y轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,6);hold on;
plot(Zshun_c,'DisplayName','Zshun','LineWidth',1);
title("补偿后陀螺仪z轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,7);hold on;
plot(Xshun(:,6)-Xshun_c,'DisplayName','\deltaXshun','LineWidth',1);
title("补偿前后陀螺仪x轴测量值差");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,8);hold on;
plot(-Yshun(:,5)-Yshun_c,'DisplayName','\deltaYshun','LineWidth',1);
title("补偿前后陀螺仪y轴测量值差");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,9);hold on;
plot(Zshun(:,4)-Zshun_c,'DisplayName','\deltaZshun','LineWidth',1);
title("补偿前后陀螺仪z轴测量值差");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;

figure(9)
subplot(3,3,1);hold on;
plot(Xni(:,6),'DisplayName','Xni','LineWidth',1);
title("补偿前陀螺仪x轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,2);hold on;
plot(-Yni(:,5),'DisplayName','Yni','LineWidth',1);
title("补偿前陀螺仪y轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,3);hold on;
plot(Zni(:,4),'DisplayName','Zni','LineWidth',1);
title("补偿前陀螺仪z轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,4);hold on;
plot(Xni_c,'DisplayName','Xni','LineWidth',1);
title("补偿后陀螺仪x轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,5);hold on;
plot(Yni_c,'DisplayName','Yni','LineWidth',1);
title("补偿后陀螺仪y轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,6);hold on;
plot(Zni_c,'DisplayName','Zni','LineWidth',1);
title("补偿后陀螺仪z轴测量值");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,7);hold on;
plot(Xni(:,6)-Xni_c,'DisplayName','\deltaXni','LineWidth',1);
title("补偿前后陀螺仪x轴测量值差");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,8);hold on;
plot(-Yni(:,5)-Yni_c,'DisplayName','\deltaYni','LineWidth',1);
title("补偿前后陀螺仪y轴测量值差");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
subplot(3,3,9);hold on;
plot(Zni(:,4)-Zni_c,'DisplayName','\deltaZni','LineWidth',1);
title("补偿前后陀螺仪z轴测量值差");
xlabel("运转时间t/10^{-2}s")
ylabel("测量值")
legend;
%% 陀螺标定
%平稳转动全过程随起始时间变化过程
for i=1:1400
    [b_x(i),delt_s_x(i)]=GYO(Xshun(:,6),Xni(:,6),830+i,2630+i,804+i,2604+i,180,18);
    [b_y(i),delt_s_y(i)]=GYO(-Yshun(:,5),-Yni(:,5),585+i,2385+i,1565+i,3365+i,180,18);
    [b_z(i),delt_s_z(i)]=GYO(Zshun(:,4),Zni(:,4),710+i,2510+i,1260+i,3060+i,180,18);
end
figure(10)
subplot(2,3,1);plot(b_x,'DisplayName','b_x','LineWidth',1)
title("陀螺标定在转动角\alpha=180°时x轴参数随时间变化");
xlabel("平稳运转时间t/10^{-2}s")
ylabel("b_x/s_x")
legend
subplot(2,3,2);plot(b_y,'DisplayName','b_y','LineWidth',1)
title("陀螺标定在转动角\alpha=180°时y轴参数随时间变化");
xlabel("平稳运转时间t/10^{-2}s")
ylabel("b_y/s_y")
legend
subplot(2,3,3);plot(b_z,'DisplayName','b_z','LineWidth',1)
title("陀螺标定在转动角\alpha=180°时z轴参数随时间变化");
xlabel("平稳运转时间t/10^{-2}s")
ylabel("b_z/s_z")
legend
subplot(2,3,4);plot(delt_s_x,'DisplayName','delt_s_x','LineWidth',1,'LineStyle','--')
title("陀螺标定在转动角\alpha=180°时x轴参数随时间变化");
xlabel("平稳运转时间t/10^{-2}s")
ylabel("b_x/s_x")
legend
subplot(2,3,5);plot(delt_s_y,'DisplayName','delt_s_y','LineWidth',1,'LineStyle','--')
title("陀螺标定在转动角\alpha=180°时y轴参数随时间变化");
xlabel("平稳运转时间t/10^{-2}s")
ylabel("b_y/s_y")
legend
subplot(2,3,6);plot(delt_s_z,'DisplayName','delt_s_z','LineWidth',1,'LineStyle','--')
title("陀螺标定在转动角\alpha=180°时z轴参数随时间变化");
xlabel("平稳运转时间t/10^{-2}s")
ylabel("b_z/s_z")
legend
%% 
%从转动开始时间，随转动角度变化过程
for i=1:360
    %考虑整个过程中的加速减速情况
    %但是由于转台本身存在不稳定性，标定结果存在波动是正常情况，但是标定结果可信度并不高
    if i<10
        t=sqrt(2*i/5);
        [b_xa(i),delt_s_xa(i)]=GYO(Xshun(:,6),Xni(:,6),524,524+int16(t*100),504,504+int16(t*100),i,t);
        [b_ya(i),delt_s_ya(i)]=GYO(-Yshun(:,5),-Yni(:,5),285,285+int16(t*100),1265,1265+int16(t*100),i,t);
        [b_za(i),delt_s_za(i)]=GYO(Zshun(:,4),Zni(:,4),410,410+int16(t*100),960,960+int16(t*100),i,t);
    end
    if i>=10&&i<=350
        t=(i-10)/10+2;
        [b_xa(i),delt_s_xa(i)]=GYO(Xshun(:,6),Xni(:,6),524,524+int16((t)*100),504,504+int16((t)*100),i,t);
        [b_ya(i),delt_s_ya(i)]=GYO(-Yshun(:,5),-Yni(:,5),285,285+int16((t)*100),1265,1265+int16((t)*100),i,t);
        [b_za(i),delt_s_za(i)]=GYO(Zshun(:,4),Zni(:,4),410,410+int16((t)*100),960,960+int16((t)*100),i,t);
    end
    if i>350
        t=2-sqrt(100-10*(i-350))/5+36;
        [b_xa(i),delt_s_xa(i)]=GYO(Xshun(:,6),Xni(:,6),524,524+int16(t*100),504,504+int16(t*100),i,t);
        [b_ya(i),delt_s_ya(i)]=GYO(-Yshun(:,5),-Yni(:,5),285,285+int16(t*100),1265,1265+int16(t*100),i,t);
        [b_za(i),delt_s_za(i)]=GYO(Zshun(:,4),Zni(:,4),410,410+int16(t*100),960,960+int16(t*100),i,t);
    end
end
figure(11)
subplot(3,1,1);plot(b_xa,'DisplayName','b_x','LineWidth',1);hold on;plot(delt_s_xa,'DisplayName','delt_s_x','LineWidth',1,'LineStyle','--')
title("陀螺标定x轴参数随转动角度\alpha变化");
xlabel("转动角度\alpha/°")
ylabel("b_x/s_x")
legend
subplot(3,1,2);plot(b_ya,'DisplayName','b_y','LineWidth',1);hold on;plot(delt_s_ya,'DisplayName','delt_s_y','LineWidth',1,'LineStyle','--')
title("陀螺标定y轴参数随转动角度\alpha变化");
xlabel("转动角度\alpha/°")
ylabel("b_y/s_y")
legend
subplot(3,1,3);plot(b_za,'DisplayName','b_z','LineWidth',1);hold on;plot(delt_s_za,'DisplayName','delt_s_z','LineWidth',1,'LineStyle','--')
title("陀螺标定z轴参数随转动角度\alpha变化");
xlabel("转动角度\alpha/°")
ylabel("b_z/s_z")
legend
%% 对准
% 已知量的定义
g_n=[0;0;g];
v_g=g_n/norm(g_n);
omiga_n_ie=[omiga_e*cos(Lat);0;-omiga_e*sin(Lat)];
v_omiga=cross(g_n,omiga_n_ie)/norm(cross(g_n,omiga_n_ie));
v_gomiga=cross(cross(g_n,omiga_n_ie),g_n)/norm(cross(cross(g_n,omiga_n_ie),g_n));
meanx=0;
meany=0;
meanz=0;
meanfx=0;
meanfy=0;
meanfz=0;
for i=1:height(Alig)
    meany=(meany*(i-1)+Alig(i,6))/i;
    meanx=(meanx*(i-1)+Alig(i,5))/i;
    meanz=(meanz*(i-1)+Alig(i,4))/i;
    meanfy=(meanfy*(i-1)+Alig(i,3))/i;
    meanfx=(meanfx*(i-1)+Alig(i,2))/i;
    meanfz=(meanfz*(i-1)+Alig(i,1))/i;
    
    %不取平均
%     meany=Alig(i,6);
%     meanx=Alig(i,5);
%     meanz=Alig(i,4);
%     meanfy=Alig(i,3);
%     meanfx=Alig(i,2);
%     meanfz=Alig(i,1);

    omiga_b_ie=[-meanx;meany;-meanz];
    g_b=-[-meanfx;meanfy;-meanfz];
    omiga_g=g_b/norm(g_b);
    omiga_omiga=cross(g_b,omiga_b_ie)/norm(cross(g_b,omiga_b_ie));
    omiga_gomiga=cross(cross(g_b,omiga_b_ie),g_b)/norm(cross(cross(g_b,omiga_b_ie),g_b));
    C_n_b=[v_g,v_omiga,v_gomiga]*[transpose(omiga_g);transpose(omiga_omiga);transpose(omiga_gomiga)];
    Theta(i,1)=atand(-C_n_b(3,1)/sqrt((C_n_b(3,2)^2)+(C_n_b(3,3)^2)));     %俯仰角/degree
    Phai(i,1)=atan2d(C_n_b(3,2),C_n_b(3,3));                               %横滚角/degree
    Pusai(i,1)=atan2d(C_n_b(2,1),C_n_b(1,1));                              %航向角/degree
end
%% 绘图
figure(12)
subplot(3,1,1);plot(Theta);
title("俯仰角\theta")
xlabel("测量时间t/10^{-2}s")
ylabel("\theta")
subplot(3,1,2);plot(Phai);
title("横滚角\phi")
xlabel("测量时间t/10^{-2}s")
ylabel("\phi")
subplot(3,1,3);plot(Pusai);
title("航向角\psi")
xlabel("测量时间t/10^{-2}s")
ylabel("\psi")
%% 陀螺标定函数
function[b,delt]=GYO(shun,ni,beg1,end1,beg2,end2,angle,t)
%[零偏，比例因子] = (顺时针数据，逆时针数据，顺时针起始时刻，顺时针终止时刻，逆时针起始时刻，逆时针终止时刻，旋转角度，旋转时间)
    Samp_rate=100;
    omiga_e= 7.292115*1e-5; 
    Lat=deg2rad(30.531651244);
    a1=0;
    a2=0;
    %进行角度积分
    for i=beg1:end1
        a1=a1+shun(i);
    end
    for i=beg2:end2
        a2=a2+ni(i);
    end
    %统一除以采样率
    a1=a1/Samp_rate;
    a2=a2/Samp_rate;
    b=(a1+a2)/(2*t)-omiga_e*sin(Lat);
    delt=((a1-a2)/(2*deg2rad(angle)))-1;
end
function[b,delt]=GYO1(up,down)
    omiga_e= 7.292115*1e-5; 
    Lat=deg2rad(30.531651244);
    meanshun=mean(up);
    meanni=mean(down);
    b=(meanshun+meanni)/2;
    delt=(meanshun-meanni)/(2*omiga_e*sin(Lat))-1;
end