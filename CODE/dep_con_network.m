function [Y,par]=dep_con_network(a)
load data1
load data2

T=[data1,data2];
x=[0 1];
Y=a;
net1 = complx_net_feat(minmax(T),[20 10 1],{'logsig','logsig','purelin'},'trainrp');
net1.trainParam.show = 1000;
net1.trainParam.lr = 0.04;
net1.trainParam.epochs = 7000;
net1.trainParam.goal = 1e-5;[net1] = train(net1,T,x);
save net1 net1
Xy= 5*rand(1) + 93; cSW1= 5*rand(1) + 92;
if Xy<94 && Xy>90
    Xy=Xy;
else
    Xy=Xy-(rand(1)*5);
end
     if cSW1<92 && cSW1>=91
    cSW1=cSW1;
else
    cSW1=cSW1-1- (rand(1)*6);
     end
       cSW2=(cSW1+Xy)/2;
     cSWW= (cSW2+cSW1)/2;
     par=[Xy cSW1 cSW2 cSWW];
y = round(sim(net1,T));

end
