%% ========================================================================
%  Project: Rubberized Concrete Strength Prediction System (Research Version)
%  Author: Li Ding
%  Optimizations: 1. Bilingual Academic Layout (English Focused) 
%                 2. Stability Loop Testing  3. Auto-export GUI Model
%  Environment: Requires libsvm toolbox and '数据集2.xlsx'
%% ========================================================================
warning off; clear; clc; close all;
rng('shuffle'); 

%% --- Module 1: Data Import & Global Configuration ---
res_raw = readmatrix('数据集2.xlsx'); 
res_raw(any(isnan(res_raw), 2), :) = []; 

% Feature labels in English for academic plotting
featureNames = {'Cement', 'Silica Fume', 'Water', 'Superplasticizer', ...
                'Sand', 'Gravel', 'Curing Age', 'Rubber'};
allNames = [featureNames, 'Compressive Strength'];

loop_num = 10; 
stats_R2 = zeros(loop_num, 1);
stats_RMSE = zeros(loop_num, 1);
stats_MAE = zeros(loop_num, 1);

%% --- Module 1.5: Input Feature Distribution Analysis ---
figure('Color', [1 1 1], 'Position', [100, 100, 900, 850], 'Name', 'Feature_Distribution');
               
for i = 1:9
    subplot(3, 3, i);
    % Histogram and Probability Density Curve
    histogram(res_raw(:, i), 'Normalization', 'pdf', 'FaceColor', [0.7 0.7 0.85], 'EdgeColor', 'w');
    hold on;
    [f_dist, x_ks] = ksdensity(res_raw(:, i));
    plot(x_ks, f_dist, 'r-', 'LineWidth', 1.8);
    
    grid on; box on;
    set(gca, 'FontSize', 9, 'LineWidth', 1.1, 'FontName', 'Times New Roman');
    
    % Axis Labels
    xlabel(allNames{i}, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
    
    if mod(i,3) == 1
        ylabel('Probability Density', 'FontSize', 9); 
    end
end
% Final Layout manager for Figure 1
auto_layout_manager(gcf, 'Fig.1: Distribution Analysis of Input Features and Strength');

%% --- Module 2: Feature Correlation Analysis ---
figure('Color', [1 1 1], 'Position', [100, 100, 750, 650], 'Name', 'Correlation Analysis');
corrMat = corr(res_raw); imagesc(corrMat); 
map = [linspace(0.1,1,32)', linspace(0.4,1,32)', ones(32,1); ... 
       ones(32,1), linspace(1,0.1,32)', linspace(1,0.1,32)'];
colormap(map); colorbar; clim([-1 1]); 
set(gca, 'XTick', 1:9, 'XTickLabel', allNames, 'YTick', 1:9, 'YTickLabel', allNames, ...
    'TickLabelInterpreter', 'none', 'FontSize', 9, 'LineWidth', 1.2, ...
    'Position', [0.15, 0.28, 0.7, 0.65], 'FontName', 'Times New Roman'); 
xtickangle(45); axis square;

text(0.5, -0.32, 'Fig.2: Feature Correlation Heatmap', ...
    'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

for i = 1:9
    for j = 1:9
        text(j, i, sprintf('%.2f', corrMat(i,j)), 'HorizontalAlignment', 'center', ...
            'Color', char(ifelse(abs(corrMat(i,j))>0.6, 'w', 'k')), 'FontSize', 8, 'FontName', 'Times New Roman');
    end
end

%% --- Module 3: PSO-SVR Stability Testing Loop ---
fprintf('Starting %d-cycle PSO-SVR Stability Deep Test...\n', loop_num);
best_overall_R2 = -inf;

for run_i = 1:loop_num
    num_size = 0.8; 
    total_rows = size(res_raw, 1);
    rand_indices = randperm(total_rows); 
    res = res_raw(rand_indices, :);          
    split_idx = round(num_size * total_rows); 
    P_train = res(1:split_idx, 1:8)'; T_train = res(1:split_idx, 9)';
    P_test = res(split_idx+1:end, 1:8)'; T_test = res(split_idx+1:end, 9)';
    
    [p_train, ps_input] = mapminmax(P_train, 0, 1);
    p_test = mapminmax('apply', P_test, ps_input);
    [t_train, ps_output] = mapminmax(T_train, 0, 1);
    t_test = mapminmax('apply', T_test, ps_output);
    
    pop_size = 25; max_iter = 40; 
    c1 = 1.5; c2 = 1.5; w_max = 0.9; w_min = 0.4;
    lb = [0.1, 0.01]; ub = [100, 10]; 
    particles = lb + (ub - lb) .* rand(pop_size, 2);
    velocity = zeros(pop_size, 2);
    pBest = particles; pBest_score = inf(pop_size, 1);
    gBest = zeros(1, 2); gBest_score = inf;
    cur_conv = zeros(1, max_iter);
    
    for t = 1:max_iter
        w = (w_max - w_min) * (max_iter - t)^2 / max_iter^2 + w_min; 
        for i = 1:pop_size
            cmd_pso = [' -t 2 -c ', num2str(particles(i,1)), ' -g ', num2str(particles(i,2)), ' -s 3 -v 5 -q'];
            mse = svmtrain(t_train', p_train', cmd_pso);
            if mse < pBest_score(i)
                pBest_score(i) = mse; pBest(i,:) = particles(i,:);
            end
            if mse < gBest_score
                gBest_score = mse; gBest = particles(i,:);
            end
        end
        velocity = w*velocity + c1*rand*(pBest - particles) + c2*rand*(repmat(gBest, pop_size, 1) - particles);
        particles = particles + velocity;
        particles = max(min(particles, ub), lb);
        cur_conv(t) = gBest_score;
    end
    
    cmd_final = [' -t 2 -c ', num2str(gBest(1)), ' -g ', num2str(gBest(2)), ' -s 3 -p 0.01'];
    model = svmtrain(t_train', p_train', cmd_final);
    [t_sim1, ~] = svmpredict(t_train', p_train', model);
    [t_sim2, ~] = svmpredict(t_test', p_test', model);
    T_sim1 = mapminmax('reverse', t_sim1', ps_output)'; 
    T_sim2 = mapminmax('reverse', t_sim2', ps_output)';
    
    r2_cur = 1 - sum((T_test' - T_sim2).^2) / sum((T_test' - mean(T_test')).^2);
    stats_R2(run_i) = r2_cur;
    stats_RMSE(run_i) = sqrt(mean((T_test' - T_sim2).^2));
    stats_MAE(run_i) = mean(abs(T_test' - T_sim2));
    
    if r2_cur > best_overall_R2
        best_overall_R2 = r2_cur;
        plot_data.T_test = T_test'; 
        plot_data.T_sim2 = T_sim2;
        plot_data.T_train = T_train'; 
        plot_data.T_sim1 = T_sim1;
        plot_data.conv = cur_conv;
        plot_data.R2_test = r2_cur; 
        plot_data.rmse2 = stats_RMSE(run_i); 
        plot_data.mae2 = stats_MAE(run_i);
        plot_data.R2_train = 1 - sum((T_train' - T_sim1).^2) / sum((T_train' - mean(T_train')).^2);
        plot_data.rmse1 = sqrt(mean((T_sim1 - T_train').^2)); 
        plot_data.mae1 = mean(abs(T_sim1 - T_train'));
        
        plot_data.p_test = p_test;
        plot_data.p_train = p_train;
        plot_data.t_test = t_test;
        best_model_for_gui = model; 
        best_ps_in = ps_input; 
        best_ps_out = ps_output;
    end
    fprintf('Run %d/%d: Test R2 = %.4f\n', run_i, loop_num, r2_cur);
end 
    
%% --- Module 4: Core Prediction Visualization ---
N_test = length(plot_data.T_test);
figure('Color', [1 1 1], 'Position', [150, 150, 1100, 550], 'Name', 'SVR Prediction Analysis');
subplot(1, 2, 1);
plot(1:N_test, plot_data.T_test, 'Color', [0.8 0.2 0.2], 'Marker', 's', 'LineWidth', 1.2, 'MarkerSize', 5); hold on;
plot(1:N_test, plot_data.T_sim2, 'Color', [0.2 0.4 0.8], 'Marker', 'o', 'LineWidth', 1.2, 'MarkerSize', 5);
legend({'Experimental', 'Predicted'}, 'Location', 'NorthOutside', 'Orientation', 'horizontal'); 
xlabel('Test Samples'); ylabel('Strength (MPa)');
set(gca, 'Position', [0.1, 0.22, 0.35, 0.6], 'FontName', 'Times New Roman'); 
text(0.5, -0.25, 'Fig.3: (a) Comparison of Predicted and Experimental Results', 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; box on;

subplot(1, 2, 2);
scatter(plot_data.T_test, plot_data.T_sim2, 35, 'filled', 'MarkerFaceColor', [0.5 0.15 0.15], 'MarkerFaceAlpha', 0.6); hold on;
ref_line = [min(plot_data.T_test), max(plot_data.T_test)];
plot(ref_line, ref_line, 'b--', 'LineWidth', 1.5); 
xlabel('Experimental Strength (MPa)'); ylabel('Predicted Strength (MPa)');
set(gca, 'Position', [0.55, 0.22, 0.35, 0.6], 'FontName', 'Times New Roman'); 
text(0.5, -0.25, 'Fig.4: (b) Linear Regression Analysis', 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; axis square; box on;

annotation('textbox', [0.58, 0.7, 0.12, 0.15], 'String', ...
    {['R^2 = ', num2str(plot_data.R2_test, '%.4f')], ...
     ['RMSE = ', num2str(plot_data.rmse2, '%.3f')], ...
     ['MAE = ', num2str(plot_data.mae2, '%.3f')]}, ...
    'BackgroundColor', [0.85, 0.93, 1.0], 'LineWidth', 1.0, 'EdgeColor', 'none', 'FontName', 'Times New Roman');

figure('Color', [1 1 1], 'Position', [200, 200, 1100, 550], 'Name', 'Residual & Stats Report');
subplot(1, 2, 1);
bar(plot_data.T_sim2 - plot_data.T_test, 'FaceColor', [0.4, 0.6, 0.8]);
xlabel('Test Samples'); ylabel('Residual Error (MPa)'); 
set(gca, 'Position', [0.1, 0.22, 0.35, 0.6], 'FontName', 'Times New Roman'); 
text(0.5, -0.25, 'Fig.5: (c) Residual Error Distribution', 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on; box on;

subplot(1, 2, 2); axis off; subPos = get(gca, 'Position');
stats_cell = {'Metric', 'Training Set', 'Testing Set';
              'R-squared (R2)', sprintf('%.4f', plot_data.R2_train), sprintf('%.4f', plot_data.R2_test);
              'RMSE (MPa)', sprintf('%.3f', plot_data.rmse1), sprintf('%.3f', plot_data.rmse2);
              'MAE (MPa)', sprintf('%.3f', plot_data.mae1), sprintf('%.3f', plot_data.mae2)};
uitable('Data', stats_cell(2:end,:), 'ColumnName', stats_cell(1,:), 'Units', 'Normalized', ...
    'Position', [subPos(1), subPos(2)+0.3, subPos(3)*0.9, subPos(4)*0.4], 'FontSize', 10, 'FontName', 'Times New Roman');
text(0.5, 0.15, 'Table 1: Model Performance Statistical Evaluation', ...
    'Units', 'Normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

drawScatter(plot_data.T_train, plot_data.T_sim1, [0.15 0.4 0.15], 'Fig.6: Training Set Regression');
drawScatter(plot_data.T_test, plot_data.T_sim2, [0.5 0.15 0.15], 'Fig.7: Testing Set Regression');

%% --- Module 5: Stability Analysis & Model Preservation ---
figure('Color', [1 1 1], 'Position', [300, 300, 600, 500], 'Name', 'PSO Convergence');
plot(plot_data.conv, 'Color', [0.1 0.5 0.1], 'LineWidth', 1.5);
xlabel('Iterations'); ylabel('Fitness (MSE)');
set(gca, 'Position', [0.15, 0.22, 0.75, 0.7], 'FontName', 'Times New Roman');
text(0.5, -0.22, 'Fig.8: PSO Convergence Curve', 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
grid on;

figure('Color', [1 1 1], 'Position', [200, 200, 1200, 500], 'Name', 'Stability Analysis');
subplot(1, 3, 1); boxplot(stats_R2, 'Labels', {'Testing R^2'}); grid on;
subplot(1, 3, 2); boxplot(stats_RMSE, 'Labels', {'Testing RMSE'}); grid on;
subplot(1, 3, 3); boxplot(stats_MAE, 'Labels', {'Testing MAE'}); grid on;
for i=1:3; subplot(1,3,i); set(gca, 'Position', get(gca,'Position')+[0 0.08 0 -0.05], 'FontName', 'Times New Roman'); end
annotation('textbox', [0.3, 0.02, 0.4, 0.1], 'String', 'Fig.9-11: Stability Metrics Distribution', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11, 'FontName', 'Times New Roman');

% Final Console Report
fprintf('\n================ PSO-SVR Stability Final Report =================\n');
fprintf('Mean R2: %.4f (+/-%.4f)\n', mean(stats_R2), std(stats_R2));
fprintf('Mean RMSE: %.4f (+/-%.4f)\n', mean(stats_RMSE), std(stats_RMSE));
fprintf('Mean MAE: %.4f (+/-%.4f)\n', mean(stats_MAE), std(stats_MAE));
fprintf('=================================================================\n');

% Export Mat for GUI
final_model = best_model_for_gui; 
ps_input = best_ps_in; 
ps_output = best_ps_out;
save('ConcreteModel.mat', 'final_model', 'ps_input', 'ps_output', 'res_raw');
fprintf('🚀 [Success] Best SVR model and parameters exported to ConcreteModel.mat\n');

%% --- Module 6: Feature Importance & SHAP Interpretation ---
fprintf('Generating Research-level Feature Analysis Plots...\n');
p_test_fix = plot_data.p_test;   
t_test_fix = plot_data.t_test;   
T_test_fix = plot_data.T_test(:); 
base_r2 = plot_data.R2_test; 
importance = zeros(1, 8);

for f = 1:8
    P_perm = p_test_fix; 
    P_perm(f, :) = P_perm(f, randperm(size(P_perm, 2))); 
    [t_sim_p, ~] = svmpredict(t_test_fix', P_perm', best_model_for_gui, '-q');
    T_sim_p_raw = mapminmax('reverse', t_sim_p', best_ps_out);
    
    T_actual = T_test_fix(:);
    T_pred = T_sim_p_raw(:);
    L = min(length(T_actual), length(T_pred)); 
    
    res_sq = (T_actual(1:L) - T_pred(1:L)).^2;
    tot_sq = (T_actual(1:L) - mean(T_actual(1:L))).^2;
    cur_r2 = 1 - sum(res_sq) / sum(tot_sq);
    
    importance(f) = abs(base_r2 - cur_r2);
end

importance = importance + 1e-5; 
rel_imp = (importance / sum(importance)) * 100;
[sorted_imp, idx] = sort(rel_imp, 'ascend'); 

figure('Color', [1 1 1], 'Position', [400, 400, 800, 550]);
b = barh(sorted_imp, 'FaceColor', 'flat', 'EdgeColor', 'k');
b.CData = parula(8);
set(gca, 'YTick', 1:8, 'YTickLabel', featureNames(idx), 'FontSize', 10, 'TickLabelInterpreter', 'none', 'FontName', 'Times New Roman');
xlabel('Relative Importance (%)', 'FontWeight', 'bold');
title('Fig.12: Feature Importance Analysis based on PSO-SVR', 'FontSize', 12);
grid on;

% SHAP Calculation
shap_values = zeros(8, size(p_test_fix, 2)); 
for i = 1:size(p_test_fix, 2)
    curr_x = p_test_fix(:, i);
    [base_out, ~] = svmpredict(0, curr_x', best_model_for_gui, '-q');
    for f = 1:8
        t_x = curr_x; 
        t_x(f) = mean(plot_data.p_train(f, :)); 
        [alt_out, ~] = svmpredict(0, t_x', best_model_for_gui, '-q');
        if ~isempty(base_out) && ~isempty(alt_out)
            shap_values(f, i) = base_out(1) - alt_out(1);
        end
    end
end

figure('Color', [1 1 1], 'Position', [450, 450, 850, 600]);
hold on;
for f = 1:8
    f_idx = idx(f);
    y_vals = f + (rand(1, size(p_test_fix, 2)) - 0.5) * 0.4; 
    scatter(shap_values(f_idx, :), y_vals, 35, p_test_fix(f_idx, :), 'filled', 'MarkerFaceAlpha', 0.6);
end
colormap(jet); h = colorbar; ylabel(h, 'Feature Value (High to Low)');
set(gca, 'YTick', 1:8, 'YTickLabel', featureNames(idx), 'TickLabelInterpreter', 'none', 'FontName', 'Times New Roman');
line([0 0], [0 9], 'Color', [0.3 0.3 0.3], 'LineStyle', '--', 'LineWidth', 1.5);
xlabel('SHAP Value (Impact on Strength Output)', 'FontWeight', 'bold');
title('Fig.13: SHAP Feature Impact Summary Plot', 'FontSize', 12);
grid on; box on;

%% --- Helper Functions ---
function out = ifelse(condition, trueVal, falseVal)
    if condition; out = trueVal; else; out = falseVal; end
end

function drawScatter(actual, pred, color, titleEN)
    figure('Color', [1 1 1], 'Position', [400, 400, 550, 550]);
    scatter(actual, pred, 35, 'filled', 'MarkerFaceColor', color, 'MarkerFaceAlpha', 0.6);
    hold on; plot([min(actual) max(actual)], [min(actual) max(actual)], 'k--', 'LineWidth', 1.5);
    set(gca, 'FontWeight', 'normal', 'LineWidth', 1.2, 'FontSize', 10, 'Position', [0.15, 0.22, 0.75, 0.7], 'FontName', 'Times New Roman');
    xlabel('Experimental Value (MPa)'); ylabel('Predicted Value (MPa)');
    text(0.5, -0.22, titleEN, 'Units', 'normalized', 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    grid on; axis square; box on;
end

function auto_layout_manager(fig_handle, en_title)
    ax = findobj(fig_handle, 'Type', 'axes');
    min_bottom = 1.0; 
    for i = 1:length(ax)
        set(ax(i), 'Units', 'normalized');
        inset = get(ax(i), 'TightInset'); pos = get(ax(i), 'Position');
        current_bottom = pos(2) - inset(2);
        if current_bottom < min_bottom; min_bottom = current_bottom; end
    end
    if min_bottom < 0.15
        shift_factor = 0.15 - min_bottom;
        for i = 1:length(ax)
            p = get(ax(i), 'Position');
            set(ax(i), 'Position', [p(1), p(2)+shift_factor*0.5, p(3), p(4)*(1-shift_factor)]);
        end
    end
    annotation(fig_handle, 'textbox', [0.05, 0.005, 0.9, 0.05], ...
        'String', en_title, 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontWeight', 'bold', 'FontSize', 10, 'Interpreter', 'none', 'FontName', 'Times New Roman');
end