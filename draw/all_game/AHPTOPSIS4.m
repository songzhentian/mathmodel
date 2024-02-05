function result=AHPTOPSIS4(match,pl)

st2=table();
if pl==1
    gd_st=match(:,3)-min(match(:,3));
    st2=addvars(st2,gd_st);

    fd_st=match(:,4)-min(match(:,4));
    st2=addvars(st2,fd_st);
    
    p1_acest=match(:,5);
    st2=addvars(st2,p1_acest);
    
    p1_faultst=max(match(:,7))-match(:,7);
    st2=addvars(st2,p1_faultst);

    p1_dist=max(match(:,9))-match(:,9);
    st2=addvars(st2,p1_dist);

    ser_st=[];
    for i=1:length(match)
        if match(i,11)==1
            ser_st(i)=1;
        end
        if match(i,11)==0
            ser_st(i)=0;
        end
    end
    ser_st=ser_st';
    st2=addvars(st2,ser_st);

    p1_n=match(:,12);
    st2=addvars(st2,p1_n);

    p1_nw=match(:,14);
    st2=addvars(st2,p1_nw);

    p1_b=match(:,16);
    st2=addvars(st2,p1_b);

    p1_bw=match(:,18);
    st2=addvars(st2,p1_bw);

    p1_bm=max(match(:,20))-match(:,20);
    st2=addvars(st2,p1_bm);

    data=table2array(st2);
end
if pl==2
    match(:,3)=-match(:,3);
    match(:,4)=-match(:,4);
    gd_st=match(:,3)-min(match(:,3));
    st2=addvars(st2,gd_st);

    fd_st=match(:,4)-min(match(:,4));
    st2=addvars(st2,fd_st);
    
    p1_acest=match(:,6);
    st2=addvars(st2,p1_acest);
    
    p1_faultst=max(match(:,8))-match(:,8);
    st2=addvars(st2,p1_faultst);

    p1_dist=max(match(:,10))-match(:,10);
    st2=addvars(st2,p1_dist);

    ser_st=[];
    for i=1:length(match)
        if match(i,11)==1
            ser_st(i)=1;
        end
        if match(i,11)==0
            ser_st(i)=0;
        end
    end
    ser_st=ser_st';
    st2=addvars(st2,ser_st);

    p1_n=match(:,13);
    st2=addvars(st2,p1_n);

    p1_nw=match(:,15);
    st2=addvars(st2,p1_nw);

    p1_b=match(:,17);
    st2=addvars(st2,p1_b);

    p1_bw=match(:,19);
    st2=addvars(st2,p1_bw);

    p1_bm=max(match(:,21))-match(:,21);
    st2=addvars(st2,p1_bm);


    data=table2array(st2);
end

SumSqr=sumsqr(data);
result=zeros(height(data),1);
data=data/sqrt(SumSqr);
temp_data=data;
for j=1:width(data)
    if find(data(:,j)<0)
        temp_data(:,j)=(data(:,j)-min(data(:,j)))/(max(data(:,j))-min(data(:,j)));
    end
end
[m, n] = size(temp_data);
p=zeros(m,n);
for j=1:width(data)
    p(:,j)=temp_data(:,j)/sum(temp_data(:,j))+0.000000000001;
end
e=zeros(width(data),1);
for j=1:width(data)
    sum1=0;
    for i=1:height(data)
        sum1=sum1+p(i,j)*log(p(i,j));
    end
    e(j)=-sum1/log(height(data));
end
d=1-e;
w=d/sum(d);

Maxz=max(data);
Minz=min(data);
Dz=zeros(height(data));
Df=zeros(height(data));
for i=1:height(data)
    for j=1:width(data)
        Dz(i)=Dz(i)+w(j)*(data(i,j)-Maxz(j))^2;
        Df(i)=Df(i)+w(j)*(data(i,j)-Minz(j))^2;
    end
    Dz(i)=sqrt(Dz(i));
    Df(i)=sqrt(Df(i));
    result(i)=Df(i)/(Dz(i)+Df(i));
end
result=result./sum(result);

end