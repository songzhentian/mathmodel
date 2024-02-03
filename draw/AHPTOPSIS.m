function result=AHPTOPSIS(match,pl)

st=table();
if pl==1
    gd_st=match.g_differ-min(match.g_differ);
    st=addvars(st,gd_st);

    st1=match.p_differ-min(match.p_differ);
    st=addvars(st,st1);
    
    p1_acest=match{:,"p1_ace"};
    st=addvars(st,p1_acest);
    
    p1_faultst=max(match.p1_fault)-match.p1_fault;
    st=addvars(st,p1_faultst);

    p1_dist=max(match.p1_dis)-match.p1_dis;
    st=addvars(st,p1_dist);

%     ser_st=[];
%     for i=1:height(match)
%         if match{i,"server"}==1
%             ser_st(i)=3;
%         end
%         if match{i,"server"}==2
%             ser_st(i)=0;
%         end
%     end
%     ser_st=ser_st';
%     st=addvars(st,ser_st);
    data=table2array(st);
end
if pl==2
    match{:,1}=-match{:,1};
    match{:,2}=-match{:,2};
    gd_st=match.g_differ-min(match.g_differ);
    st=addvars(st,gd_st);

    st1=match.p_differ-min(match.p_differ);
    st=addvars(st,st1);
    
    p2_acest=match{:,"p2_ace"};
    st=addvars(st,p2_acest);
    
    p2_faultst=max(match.p2_fault)-match.p2_fault;
    st=addvars(st,p2_faultst);

    p2_dist=max(match.p2_dis)-match.p2_dis;
    st=addvars(st,p2_dist);

%     ser_st=[];
%     for i=1:height(match)
%         if match{i,"server"}==2
%             ser_st(i)=3;
%         end
%         if match{i,"server"}==1
%             ser_st(i)=0;
%         end
%     end
%     ser_st=ser_st';
%     st=addvars(st,ser_st);
    data=table2array(st);
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