clear all; 

fmae = strcat('LP/mae01MP.dat');
dlmwrite(fmae, []);

for i=10:10:100
    fA = strcat('LP/A', num2str(i), '.dat');
    fb = strcat('LP/b', num2str(i), '.dat');
    fD = strcat('LP/D', num2str(i), '.dat');
    
    A = load(fA);
    b = load(fb);
    D = load(fD);
    
    n = D(1); % total # of nodes in non-sparsified graph
    s = D(2); % sum of discreapncies of sparsified nodes

    A1 = spconvert(A);
    d = spconvert(b); % expected degrees of non-sparsified nodes

    m = size(A1,2);

    f = -ones(1,m);

    tic
    [x, fval] = linprog(f, A1, d, [], [], zeros(m,1), ones(m,1));
    toc 

    a = sum(abs(d - A1*x)); % sum of discreapncies of non-sparsified nodes
    mae = (s + a)/n;
    
    dlmwrite(fmae, [i/100, mae], '-append');
end