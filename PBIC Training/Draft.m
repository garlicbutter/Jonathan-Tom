a=[];
a(1)=0;
for i=1:1:100
    a(i+1) = a(i)+1;
    save('a.mat','a');
end
