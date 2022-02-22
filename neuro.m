clear;

%% Calculate ApEn matrix
data_dir = dir(pwd);
% pwd folder should be 'F:\Aseem_IITD\ELL319_TP\data'
n = 1098; % num of datapoints
apEn = zeros(1098, 160);

for k = 4:length(data_dir)
    sprintf("working on index %d", k-3)
    sub_dir = data_dir(k).name;
    if isfolder(sub_dir)
        sub_file = dir(fullfile(sub_dir, '*.mat'));
        % age can be read here as well
        sub_data = load(fullfile(sub_dir, sub_file.name));
        data = sub_data.y_roi_regressed_wobadTP;
    end
    
    for m = 1:160
        apEn(k-3, m) = approximateEntropy(data(:, m));
    end
end

%% Read the age matrix separately 
age = readmatrix('..\age_final.csv');

%% Find optimum cluster number
% eval = evalclusters(apEn, 'kmeans', 'silhouette', 'KList', 1:6);
% above is MATLAB built in and gives optimal as k = 3

% below copied from -- https://stackoverflow.com/questions/46473719/optimum-number-of-clusters-in-k-mean-clustering-using-bic-matlab
BIC = []; % Bayesian Information Criterion 
for k=1:10  % number of clusters
    RSS=0;  % residual sum of squares
    [idx, C] = kmeans(apEn, k); 
    
    for i = 1:n
        RSS = RSS + sqrt((apEn(i,1) - C(idx(i),1))^2 + (apEn(i,2) - C(idx(i),2))^2);
    end
    BIC(k) = n*log(RSS/n)+(k*3)*log(n);
end
[min_BIC, k_opt] = min(BIC);
plot(BIC)
% above gives 7 but should be 4, needs minor corrections

%% Perform kmeans on k_opt
idx = kmeans(apEn, k_opt);

%% to find age groups from optimal clusters
lower = zeros(k_opt);
upper = zeros(k_opt);

for k = 1:n
    if age(k) > upper(idx(k))
        upper(idx(k)) = age(k)
    end
    if age(k) < lower(idx(k))
        lower(idx(k)) = age(k)
    end
end

for k = 1:k_opt
    sprintf("age boundary %d: %f to %f", k, lower(k), upper(k))
end  