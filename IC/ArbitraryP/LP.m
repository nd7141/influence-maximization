clear all; 

for i=10:10:100
    fA = strcat('LP/A', num2str(i), '.dat');
    fb = strcat('LP/b', num2str(i), '.dat');
    
    A = load(fA);
    b = load(fb);
    
    A1 = spconvert(A);
    d = spconvert(b);
    
    size(A1)
    
    m = size(A1,2);

    f = -ones(1,m);

    [x, fval] = linprog(f, A1, d, [], [], zeros(m,1), ones(m,1));
    
    fx = strcat('LP/x', num2str(i), '.txt');
    dlmwrite(fx, x);
end