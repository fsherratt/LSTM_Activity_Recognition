% Load UCI-HAD and convert to usable format for tensorflow 2.0
ROOT_DIR = 'Original_Data\HAPT Data Set\RawData\';
SAVE_DIR = 'Transform_Data\';

disp '-- Running --'

%% Load list of files
acc_file_list   = dir([ROOT_DIR, 'acc',   '*.txt']);
gyro_file_list  = dir([ROOT_DIR, 'gyro',  '*.txt']);
label_file_list = dir([ROOT_DIR, 'label', '*.txt']);

%% Load File Data
filename = label_file_list(1).name;

activity_names = { 'UNSPECIFIED', ... %0
               'WALKING', ... %1
              'WALKING_UPSTAIRS', ... %2 
              'WALKING_DOWNSTAIRS', ... %3
              'SITTING', ... %4
              'STANDING', ... %5
              'LAYING', ... %6
              'STAND_TO_SIT', ... %7
              'SIT_TO_STAND', ... %8
              'SIT_TO_LIE', ... %9
              'LIE_TO_SIT', ... %10
              'STAND_TO_LIE', ... %11
              'LIE_TO_STAND'}; %12
          
fileID = fopen([ROOT_DIR, filename],'r');
dataArray = textscan(fileID, '%f%f%f%f%f%[^\n\r]', 'Delimiter', ' ', 'MultipleDelimsAsOne', true, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);

label_experiment = dataArray{:, 1};
% label_user = dataArray{:, 2};
label_activity = dataArray{:, 3};
label_start = dataArray{:, 4};
label_end = dataArray{:, 5};

label_exp_end_row = find( label_experiment(2:end) - label_experiment(1:end-1) > 0 );
label_exp_start_row = [1; label_exp_end_row+1];
label_exp_end_row = [label_exp_end_row; length(label_experiment)];

clearvars filename fileID dataArray;

%% Load raw data
disp '-- Loading Data --'

data_acc = loadFileData(ROOT_DIR, acc_file_list);
data_gyro = loadFileData(ROOT_DIR, gyro_file_list);

disp '-- Data Loaded --'

%% Convert to output vector
disp '-- Processing Labels --'

labels = cell(1, length(data));

for i = 1:length(data)
    fprintf('Processing label exp: %03d\n', i);
    
    tmp_experiment_rows = label_exp_start_row(i):label_exp_end_row(i);
    
    tmp_activity = label_activity(tmp_experiment_rows);
    tmp_start = label_start(tmp_experiment_rows);
    tmp_end = label_end(tmp_experiment_rows);
    
    tmp_labels = zeros(length(data{i}), 1);
    
    for j = 1:length(tmp_activity)
        tmp_elements = tmp_start(j):tmp_end(j);
        tmp_labels(tmp_elements) = tmp_labels(tmp_elements) + tmp_activity(j);
    end
    
    labels{i} = tmp_labels;
end

clear i j tmp_* label_*

disp '-- Labels Loaded --'

%% Convert to input vectors
disp '-- Combining Data --'

data = cell(1, length(data_acc));

for i = 1:length(data_acc)
    data{i} = [labels{i}, data_acc{i}, data_gyro{i}];
end

clear labels data_acc data_gyro;


%% Save Data
disp '-- Saving Data --'

data_columns = {'Activity', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'};

for i = 1:length(data)
    filename = sprintf('exp%03d.csv', i);
    
    fprintf('Saving file data_%s\n', filename');
    T = array2table(data{i},'VariableNames',data_columns);
    writetable(T, [SAVE_DIR, 'data_', filename]);
end

clear data_columns T

disp '-- Data Saved --'

%% Clear up
clear *_DIR *_file_list ans i;

disp '-- All Done --'

%% Functions
function data = loadFileData( root_dir, file_list )
    data = cell(1, length(file_list));
    
    for i = 1:length(file_list)
        filename = file_list(i).name;

        tmp = regexpi(filename, '^(?<type>)[a-z]{0,4}_exp(?<exp>[0-9]{2})_user(?<user>[0-9]{2})', 'names','lineanchors');
        experiment = str2double(tmp.exp);
        
        fprintf('Processing data file %s\n', filename);

        formatSpec = '%f%f%f%[^\n\r]';
        delimiter = ' ';

        fileID = fopen([root_dir, filename],'r');
        dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'TextType', 'string', 'EmptyValue', NaN,  'ReturnOnError', false);
        fclose(fileID);

        data{experiment} = [dataArray{1:end-1}];
        
    end
end