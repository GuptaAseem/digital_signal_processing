clear;

%% Setup
data_dir = dir(pwd);
% pwd folder should be 'F:\Aseem_IITD\ELL319_TP\data'
n = 1098; % num of datapoints

%% Calculate the ApEn matrix (use only once to generate file)
% apEn = zeros(1098, 160);
% for k = 4:length(data_dir)
%     sprintf("working on index %d", k-3)
%     sub_dir = data_dir(k).name;
%     if isfolder(sub_dir)
%         sub_file = dir(fullfile(sub_dir, '*.mat'));
%         % age can be read here as well
%         sub_data = load(fullfile(sub_dir, sub_file.name));
%         data = sub_data.y_roi_regressed_wobadTP;
%     end
%     
%     for m = 1:160
%         apEn(k-3, m) = approximateEntropy(data(:, m));
%     end
% end
% save('..\apEn.mat', 'apEn')

%% Directly read the ApEn matrix created above
apEn_str = load('..\apEn.mat');
apEn = apEn_str.apEn;
% Read the age matrix separately 
age = readmatrix('..\age_final.csv');

% Combine age and ApEn for clustering data
% include age data
kmean_data = [apEn transpose(age)];

% % sort according to age
% [~, idx] = sort(kmean_data(:, end));
% kmean_data = make_data(idx, :);

%% Find optimum cluster number
rng default;
% % below from -- https://stackoverflow.com/questions/46473719/optimum-number-of-clusters-in-k-mean-clustering-using-bic-matlab
% BIC = []; % Bayesian Information Criterion 
% for k=1:8  % number of clusters
%     RSS=0;  % residual sum of squares
%     [idx, C] = kmeans(kmean_data, k); 
%     
%     for i = 1:n
%         RSS = RSS + sqrt((kmean_data(i,1) - C(idx(i),1))^2 + (kmean_data(i,2) - C(idx(i),2))^2);
%     end
%     BIC(k) = n*log(RSS/n)+(k*3)*log(n);
% end
% [min_BIC, k_opt] = min(BIC);
% plot(BIC)
% % above gives 7 but should be 4, needs minor corrections
% % after age answer becomes even more incorrect

% Elbow Method
% Reference: https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
inertia = zeros(8, 1);
for k = 2:9
    [idx, C, sumd, D] = kmeans(kmean_data, k);
    for j = 1:n
        inertia(k - 1) = inertia(k - 1) + D(j, idx(j)) * D(j, idx(j));
    end
end
inertia = inertia/n;
plot(2:9 ,inertia, '-o');
xlabel('k value')
ylabel('inertia')
title('Elbow method to determine optimal cluster size')

eval = evalclusters(kmean_data, 'kmeans', 'silhouette', 'KList', 3:9);
% above is MATLAB built in and gives optimal as k = 4/5

k_opt = 4; % determined visually from the graph
% Optimal value is k = 4

%% Perform kmeans on k_opt
idx = kmeans(kmean_data, k_opt);
% scatter(apEn, age);
%% to find age groups from optimal clusters
lower = Inf(k_opt);
upper = zeros(k_opt);

for k = 1:n
    if age(k) > upper(idx(k))
        upper(idx(k)) = age(k);
    end
    if age(k) < lower(idx(k))
        lower(idx(k)) = age(k);
    end
end

for k = 1:k_opt
    sprintf("age boundary %d: %f to %f", k, lower(k), upper(k))
end

%% 
