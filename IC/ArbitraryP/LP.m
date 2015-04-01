clear all; 

%    fmae = strcat('Flickr/mae01MP.dat');
%    dlmwrite(fmae, []);

for i=10:10:100
    fA = strcat('Flickr2/A', num2str(i), '.dat');
    fb = strcat('Flickr2/b', num2str(i), '.dat');
    fD = strcat('Flickr2/D', num2str(i), '.txt');
    
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
    mae = (s + a)/n

    fprob = strcat('Flickr2/x', num2str(i), '.dat');
    dlmwrite(fprob, x);
    
%     dlmwrite(fmae, [i/100, mae], '-append');
end