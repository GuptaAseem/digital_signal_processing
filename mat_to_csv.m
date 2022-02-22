data_dir = dir(pwd);

for k = 4:length(data_dir)
    sub_dir = data_dir(k).name;
    if isfolder(sub_dir)
        sub_file = dir(fullfile(sub_dir, '*.mat'));
        sub_data = load(fullfile(sub_dir, sub_file.name));
    end
    csvwrite(fullfile('..\converted', strcat(sub_dir, '.csv')), sub_data.y_roi_regressed_wobadTP);
end