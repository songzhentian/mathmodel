function result=AHPTOPSIS(data)
    SumSqr=sumsqr(data);
    result=zeros(height(data));
    data=data/SumSqr;
    temp_data=data;
    for j=1:width(data)
        if find(data(:,j)<0)
            temp_data(:,j)=(data(:,j)-min(data(:,j)))/(max(data(:,j))-min(data(:,j)));
        end
    end
    p=zeros(temp_data);
    for j=1:width(data)
        p(:,j)=temp_data(:,j)/sum(temp_data(:,j));
    end
    e=zeros(width(data));
    for j=1:width(data)
        sum1=0;
        for i=1:height(data)
            sum1=sum1+p(i,j)*ln(p(i,j));
        end
        e(j)=-sum1/ln(height(data));
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
end