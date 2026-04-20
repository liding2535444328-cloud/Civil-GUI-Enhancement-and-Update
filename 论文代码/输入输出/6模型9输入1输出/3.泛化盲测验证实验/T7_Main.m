%% ========================================================================
%  T7 Master Control Platform: SCI Multi-Model Integration System (V95)
%  Features: PNG+SVG Export, Breakpoint Resume, 3x2 Aggregated Matrices
%  AGGREGATED: Stability, Scatters, Residuals, Importance, SHAP, Time Audit
%% ========================================================================
warning off; clc; close all;
rng('shuffle');

% --- Module 7.1: Start/Resume Decision Logic ---
checkpoint_file = 'System_Checkpoint.mat';
start_idx = 1;
if exist(checkpoint_file, 'file')
    choice = questdlg('Detecting previous session. How to proceed?', ...
        'System Controller', 'Resume from Breakpoint', 'Start from Scratch', 'Resume from Breakpoint');
    if strcmp(choice, 'Resume from Breakpoint')
        fprintf('\n🔄 [RESUME] Loading stored data. Resuming from model %d...\n', 1); % Placeholder for log
        load(checkpoint_file);
        % --- 在 T7_Main.m 约 18 行 load(checkpoint_file); 之后插入 ---
    if strcmp(choice, 'Resume from Breakpoint')
        load(checkpoint_file);
        % --- 在 T7_Main.m 约 21 行 load(checkpoint_file); 之后插入 ---
        if ~exist('Best_Models_BP_Cell', 'var')
            Best_Models_BP_Cell = cell(6, 1); 
        end
% --- 保持后续代码不变 ---

        start_idx = m_idx_saved + 1;
        fprintf('🔄 [RESUME] Progress loaded. Resuming from Model %d...\n', start_idx);
    else
        fprintf('\n🗑️ [CLEAN] Deleting old checkpoint. Starting fresh...\n');
        delete(checkpoint_file);
    end
end

if start_idx == 1
    fprintf('\n🚀 [START] Initializing new evaluation session...\n');
    res_raw = readmatrix('数据集3.xlsx');
    res_raw(any(isnan(res_raw), 2), :) = []; 
    modelNames = {'PSO-SVR', 'FA-RF', 'PSO-LSBoost', 'GA-BP', 'PSO-LSSVM', 'LSTM'};
    colors = [0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19; 0.49 0.18 0.56; 0.93 0.69 0.13; 1.00 0.00 0.00];
    
    % Global Storage Initialization (约第 43行)此行为43行
    All_R2_Loop = zeros(10, 6);
    All_RMSE_Loop = zeros(10, 6);
    All_MAE_Loop = zeros(10, 6);
    Scatter_Collection = cell(6, 2); 
    Importance_Collection = cell(6, 1);
    SHAP_Collection = cell(6, 1); 
    Summary_Time = zeros(6, 1);
    
    % --- 在 T7_Main.m 第 52 行 Summary_Time = zeros(6, 1); 之后插入 ---
    
    % [核心新增] 存储 6 个模型各自的归一化参数和模型实体，用于盲测同步
    Best_Models_BP_Cell = cell(6, 1);
end

main_total_tic = tic;

%% --- Module 7.2: Execution Loop with Breakpoint Protection ---
for m_idx = start_idx:6
    curr_model = modelNames{m_idx};
    fprintf('\n============================================================\n');
    fprintf('▶️ NOW RUNNING MODEL %d/6: [%s]...\n', m_idx, curr_model);
    fprintf('------------------------------------------------------------\n');
    
   % --- Execute Individual Model Functions (约第 57 行) ---此行为57行
    switch m_idx
        case 1, [S_Data, Stats, Best_M] = T1_SVR(res_raw);
        case 2, [S_Data, Stats, Best_M] = T2_RF(res_raw);
        case 3, [S_Data, Stats, Best_M] = T3_LSBoost(res_raw);
        case 4, [S_Data, Stats, Best_M] = T4_GABP(res_raw);
        case 5, [S_Data, Stats, Best_M] = T5_LSSVM(res_raw);
        case 6, [S_Data, Stats, Best_M] = T6_LSTM(res_raw);
    end
  % --- 在 T7_Main.m 第 65 行 end (switch的结束) 之后插入 ---
    
    % [核心防护] 将当前模型及其专属比例尺存入单元格容器
    Best_Models_BP_Cell{m_idx} = Stats; 
    Best_Models_BP_Cell{m_idx}.model = Best_M; 
    % 兼容性补全：确保 SVR 和 RF 虽然没显式比例尺，但也拥有基本结构
    if m_idx <= 2 
        Best_Models_BP_Cell{m_idx}.is_builtin_scaling = true; 
    else
        Best_Models_BP_Cell{m_idx}.is_builtin_scaling = false;
    end

    % --- Synchronize Loop Data to Global Arrays ---
    All_R2_Loop(:, m_idx)   = double(Stats.R2_test_loop(:));
    All_RMSE_Loop(:, m_idx) = double(Stats.RMSE_test_loop(:));
    All_MAE_Loop(:, m_idx)  = double(Stats.MAE_test_loop(:));
    Summary_Time(m_idx)     = Stats.Time;
    
    Scatter_Collection{m_idx, 1} = double(S_Data.te_real(:));
    Scatter_Collection{m_idx, 2} = double(S_Data.te_sim(:));
    
    % Collect mechanism analysis data (Ensure T1-T6 return these fields)
    if isfield(Stats, 'importance'), Importance_Collection{m_idx} = Stats.importance; end
    if isfield(Stats, 'shap_v'), SHAP_Collection{m_idx} = Stats.shap_v; end

    % --- Real-time Progress Monitor ---
    fprintf('\n✅ [%s] COMPLETED.\n', curr_model);
    fprintf('   Mean Accuracy R2 : %.4f\n', mean(All_R2_Loop(:, m_idx)));
    fprintf('   Mean RMSE Error  : %.3f MPa\n', mean(All_RMSE_Loop(:, m_idx)));
    fprintf('   Computational Time: %.2f sec\n', Summary_Time(m_idx));
    % --- [新增] T1-T6 单体模型图片全格式保存 (png, svg, fig) ---
    fprintf('💾 Saving individual plots for [%s]...\n', curr_model);
    model_res_dir = fullfile(pwd, [curr_model, '_Results_Full']);
    if ~exist(model_res_dir, 'dir'); mkdir(model_res_dir); end
    
    m_figs = findall(0, 'Type', 'figure');
    for k_f = 1:length(m_figs)
        fig_obj = m_figs(k_f);
        f_title = get(fig_obj, 'Name');
        if contains(f_title, curr_model) % 只保存当前模型的图，不保存Agg图
            if contains(f_title, curr_model)
            try
                exportgraphics(fig_obj, fullfile(model_res_dir, [f_title, '.png']), 'Resolution', 300);
                saveas(fig_obj, fullfile(model_res_dir, [f_title, '.svg']), 'svg');
                saveas(fig_obj, fullfile(model_res_dir, [f_title, '.fig']));
            catch
                fprintf('   ⚠️ Warning: Could not save figure [%s]\n', f_title);
            end
        end
        end
    end

    % --- [新增] T3 实时精度/误差显示控制 ---
    if m_idx == 3
        fprintf('📊 PSO-LSBoost Real-time Monitor:\n');
        for r_p = 1:10
             fprintf('   Run %d: R2 = %.4f | RMSE = %.3f\n', r_p, Stats.R2_test_loop(r_p), Stats.RMSE_test_loop(r_p));
        end
    end

    % --- Save Checkpoint ---
    m_idx_saved = m_idx;
    % --- 替换 T7_Main.m 约 137 行的 save 语句 ---
   % [核心修复] 将模型实体与比例尺容器加入备份
    save(checkpoint_file, 'm_idx_saved', 'All_R2_Loop', 'All_RMSE_Loop', 'All_MAE_Loop', ...
         'Scatter_Collection', 'Importance_Collection', 'SHAP_Collection', ...
         'Summary_Time', 'modelNames', 'colors', 'res_raw', 'Best_Models_BP_Cell');

    fprintf('💾 Backup secured. Session Progress: %.1f%%\n', (m_idx/6)*100);
    close all; % Clean memory
end

% ========================================================================
%  Module 7.5: Master Integrated Blind Test (V108 Full Stats & Plot Audit)
% ========================================================================
fprintf('\n============================================================\n');
fprintf('🔍 STARTING MASTER BLIND TEST VALIDATION INTERFACE\n');
fprintf('============================================================\n');

% [核心修复] 强制定义全局特征名，杜绝负精度
featureNames_Master = {'W/B Ratio', 'Rubber Content', 'Max Particle Size', ...
                       'Cement', 'Fine Aggregate', 'Coarse Aggregate', ...
                       'SF/C Ratio', 'Superplasticizer', 'Curing Age'};

% 弹出标准对话框 (点选一次，六模联动)
[b_file, b_path] = uigetfile({'*.xlsx;*.xls;*.csv'}, 'Select Blind Test Data for Global Audit');

if isequal(b_file, 0)
    fprintf('⚠️ Blind Test skipped by user.\n');
else
    blind_full_path = fullfile(b_path, b_file);
    Blind_Global_Matrix = zeros(6, 5); % [R2, RMSE, MAE, MaxErr, AvgAcc]

    for i = 1:6
        m_name = modelNames{i};
        fprintf('>>> Model %d/6 [%s]: Syncing Scaling & Running Audit...\n', i, m_name);
        [metrics] = internal_master_blind_engine(blind_full_path, i, m_name, ...
                         featureNames_Master, Best_Models_BP_Cell{i}, res_raw);
        Blind_Global_Matrix(i, :) = metrics;
    end
    
    % --- Fig.22: 6-Model Blind Accuracy Comparison (汇总对比图) ---
    figure('Color', 'w', 'Position', [150, 150, 1100, 500], 'Name', 'Agg_Fig22_Blind_Comparison');
    subplot(1,2,1);
    b1 = bar(Blind_Global_Matrix(:,1), 'FaceColor', 'flat'); grid on;
    for k=1:6, b1.CData(k,:) = colors(k,:); end
    set(gca, 'XTick', 1:6, 'XTickLabel', modelNames, 'FontWeight', 'bold');
    ylabel('Blind Test Accuracy (R^2)'); title('Global Generalization Audit');
    subplot(1,2,2);
    bar(Blind_Global_Matrix(:,2:3), 'grouped'); grid on;
    set(gca, 'XTick', 1:6, 'XTickLabel', modelNames, 'FontWeight', 'bold');
    ylabel('Error Magnitude (MPa)'); legend({'RMSE', 'MAE'}); title('Global Error Comparison');
    auto_layout_manager_T7(gcf, 'Fig.22: Aggregated Blind Test Performance Comparison Across 6 Models', '');
    saveas(gcf, fullfile(t7_dir, 'Agg_Fig22_Blind_Comparison.png'));
end

% --- Module 7.6: AGGREGATED SCI VISUALIZATIONS (之前的对比图一个不能少) ---
% 此处包含 Fig.15 - Fig.21 的代码 (已为你整合，确保逻辑连贯)

% --- Module 7.6: AGGREGATED SCI VISUALIZATIONS (Matrix Layout 3x2) ---
t7_dir = 'T7_Final_Aggregated_Scientific_Results';
if ~exist(t7_dir, 'dir'); mkdir(t7_dir); end
featureNames = {'W/B', 'Rubber', 'Size', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};

% --- Fig.15: Accuracy Trend (Bar + Center-linked Line + Annotations) ---
figure('Color', [1 1 1], 'Position', [100, 100, 1000, 550], 'Name', 'Agg_Fig15_Accuracy_Trend');

Final_Mean_R2 = mean(All_R2_Loop)'; Final_Std_R2 = std(All_R2_Loop)';
b = bar(1:6, Final_Mean_R2, 'FaceColor', 'flat', 'BarWidth', 0.6); hold on; grid on;
for i = 1:6, b.CData(i,:) = colors(i,:); end
plot(1:6, Final_Mean_R2, '--o', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'w');
errorbar(1:6, Final_Mean_R2, Final_Std_R2, 'k', 'LineStyle', 'none', 'LineWidth', 1.1, 'CapSize', 8);
for i = 1:6, text(i, Final_Mean_R2(i)+Final_Std_R2(i)+0.01, sprintf('%.4f', Final_Mean_R2(i)), 'HorizontalAlignment', 'center', 'FontWeight', 'bold'); end
set(gca, 'XTick', 1:6, 'XTickLabel', modelNames, 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Determination Coefficient (R^2 Score)');
auto_layout_manager_T7(gcf, 'Fig.15: Comparative Analysis of Global Prediction Accuracy and Trend Analysis', '');

% --- Fig.16: RMSE and MAE Comparison (Grouped Bar + Linked Line) ---
figure('Color', [1 1 1], 'Position', [120, 120, 1000, 550], 'Name', 'Agg_Fig16_Error_Comparison');
Final_Mean_RMSE = mean(All_RMSE_Loop)'; 
Final_Mean_MAE  = mean(All_MAE_Loop)';
% 绘制分组柱状图
b_err = bar([Final_Mean_RMSE, Final_Mean_MAE], 'grouped', 'BarWidth', 0.8); hold on; grid on;
set(b_err(1), 'FaceColor', [0.2 0.4 0.6], 'EdgeColor', 'k'); % RMSE 颜色
set(b_err(2), 'FaceColor', [0.8 0.4 0.4], 'EdgeColor', 'k'); % MAE 颜色
% 计算柱状图中点坐标并绘制连接折线
x_data = zeros(6, 2);
y_data_matrix = [Final_Mean_RMSE, Final_Mean_MAE]; % 修复：先存入临时变量以支持索引
for j = 1:2
    x_data(:,j) = b_err(j).XData + b_err(j).XOffset;
    plot(x_data(:,j), y_data_matrix(:,j), '--o', 'LineWidth', 1.2, 'MarkerSize', 6, 'MarkerFaceColor', 'w');
    % 标注具体数值
    for i = 1:6
        val = y_data_matrix(i,j);
        text(x_data(i,j), val + 0.05, sprintf('%.2f', val), 'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    end
end
   set(gca, 'XTick', 1:6, 'XTickLabel', modelNames, 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Error Magnitude (MPa)');
legend({'RMSE', 'MAE'}, 'Location', 'northoutside', 'Orientation', 'horizontal');
auto_layout_manager_T7(gcf, 'Fig.16: Cross-Model Comparison of Prediction Errors (RMSE and MAE) with Trendlines', '');


% --- Fig.17: Scatter Matrix (3x2 Comparison) ---
figure('Color', [1 1 1], 'Position', [50, 50, 1300, 850], 'Name', 'Agg_Fig17_Scatters');
for i = 1:6
    subplot(2, 3, i);
    y_r = Scatter_Collection{i,1}; y_s = Scatter_Collection{i,2};
    scatter(y_r, y_s, 30, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.5); hold on;
    l_v = [min([y_r;y_s]) max([y_r;y_s])]; plot(l_v, l_v, 'k--', 'LineWidth', 1.2);
    grid on; axis square; title(modelNames{i}, 'FontSize', 12);
    xlabel('Experimental (MPa)'); ylabel('Predicted (MPa)');
    r2_tmp = 1 - sum((y_r-y_s).^2)/sum((y_r-mean(y_r)).^2);
    text(0.05, 0.9, sprintf('R^2 = %.4f', r2_tmp), 'Units', 'normalized', 'Color', 'r', 'FontWeight', 'bold');
end
auto_layout_manager_T7(gcf, 'Fig.17: Aggregated Linear Regression Scatter Distribution (3x2 Matrix)', '');

% --- Fig.18: Residual Matrix (3x2 Comparison) ---
figure('Color', [1 1 1], 'Position', [70, 70, 1300, 850], 'Name', 'Agg_Fig18_Residuals');
for i = 1:6
    subplot(2, 3, i);
    res = Scatter_Collection{i,2} - Scatter_Collection{i,1};
    bar(res, 'FaceColor', colors(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.7); hold on;
    line([0 length(res)], [0 0], 'Color', 'k', 'LineWidth', 1.1);
    grid on; title(modelNames{i}); xlabel('Sample Index'); ylabel('Error (MPa)');
end
auto_layout_manager_T7(gcf, 'Fig.18: Aggregated Residual Error Distribution and Comparison (3x2 Matrix)', '');

% --- Fig.19: Boxplot Matrix (Stability & Stats) ---
figure('Color', [1 1 1], 'Position', [90, 90, 1300, 850], 'Name', 'Agg_Fig19_Stability');
for i = 1:6
    subplot(2, 3, i);
    boxplot(All_R2_Loop(:,i), 'Colors', colors(i,:), 'Widths', 0.5); grid on;
    title(modelNames{i}, 'FontWeight', 'bold'); ylabel('R^2 Score');
    m_txt = sprintf('Mean Metrics:\nR^2: %.4f\nRMSE: %.3f\nMAE: %.3f', mean(All_R2_Loop(:,i)), mean(All_RMSE_Loop(:,i)), mean(All_MAE_Loop(:,i)));
    annotation('textbox', get(gca,'Position'), 'String', m_txt, 'FontSize', 7, 'EdgeColor', 'none', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end
auto_layout_manager_T7(gcf, 'Fig.19: Stability Evaluation and Performance Metrics Statistical Summary (3x2 Matrix)', '');

% --- Fig.20: Feature Importance Matrix ---
figure('Color', [1 1 1], 'Position', [110, 110, 1300, 850], 'Name', 'Agg_Fig20_Importance');
for i = 1:6
    subplot(2, 3, i);
    if ~isempty(Importance_Collection{i})
        [sv, si] = sort(Importance_Collection{i}, 'ascend');
        barh(sv, 'FaceColor', colors(i,:), 'EdgeColor', 'k'); 
        set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(si), 'FontSize', 8); grid on;
    end
    title(['Rank: ', modelNames{i}]); xlabel('Importance (%)');
end
auto_layout_manager_T7(gcf, 'Fig.20: Cross-Model Comparison of Feature Significant Contributions (3x2 Matrix)', '');

% --- Fig.21: Global SHAP Matrix ---
figure('Color', [1 1 1], 'Position', [130, 130, 1300, 850], 'Name', 'Agg_Fig21_SHAP');
for i = 1:6
    subplot(2, 3, i);
    if ~isempty(SHAP_Collection{i})
        sh_d = SHAP_Collection{i};
        for f = 1:9, scatter(sh_d(f,:), f+(rand(1,size(sh_d,2))-0.5)*0.3, 12, 'filled', 'MarkerFaceAlpha', 0.35); hold on; end
        line([0 0], [0 10], 'Color', 'k', 'LineStyle', '--'); set(gca, 'YTick', 1:9, 'YTickLabel', featureNames, 'FontSize', 8); grid on;
    end
    title(['Mechanism: ', modelNames{i}]); xlabel('SHAP Value');
end
auto_layout_manager_T7(gcf, 'Fig.21: Aggregated SHAP Summaries for Feature Impact Mechanism (3x2 Matrix)', '');

% --- Table 1: Final Audit Summary ---
figure('Color', [1 1 1], 'Position', [200, 200, 850, 420], 'Name', 'Agg_Table_Audit'); axis off;
t_d = cell(6, 5);
for i = 1:6, t_d(i,:) = {modelNames{i}, sprintf('%.4f', mean(All_R2_Loop(:,i))), sprintf('%.3f', mean(All_RMSE_Loop(:,i))), sprintf('%.3f', mean(All_MAE_Loop(:,i))), sprintf('%.2fs', Summary_Time(i))}; end
uitable('Data', t_d, 'ColumnName', {'Model', 'Mean R^2', 'Mean RMSE', 'Mean MAE', 'Comp. Time (s)'}, 'Units', 'Normalized', 'Position', [0.05, 0.1, 0.9, 0.8], 'FontSize', 10);
auto_layout_manager_T7(gcf, 'Table 1: Final Integrated Performance Audit and Computational Efficiency Summary', '');

% --- Module 7.7: Export & Cleanup ---
fprintf('\n>>> Exporting Aggregated scientific figures (PNG + SVG + FIG)...\n');
all_figs_agg = findall(0, 'Type', 'figure');
for k = 1:length(all_figs_agg)
    if isvalid(all_figs_agg(k))
        f_n = get(all_figs_agg(k), 'Name');
        if contains(f_n, 'Agg_')
            exportgraphics(all_figs_agg(k), fullfile(t7_dir, [f_n, '.png']), 'Resolution', 300);
            saveas(all_figs_agg(k), fullfile(t7_dir, [f_n, '.svg']), 'svg');
            saveas(all_figs_agg(k), fullfile(t7_dir, [f_n, '.fig']));
        end
    end
end
if exist(checkpoint_file, 'file'), delete(checkpoint_file); end
% --- 粘贴开始位置：紧跟在 delete(checkpoint_file) 之后 ---
final_total_time = toc(main_total_tic);
fprintf('\n🏆 [COMPLETED] Integrated session finished.\n');
fprintf('   Total Session Runtime: %.2f seconds.\n', final_total_time);
% ========================================================================
%  BELOW ARE FUNCTION DEFINITIONS (必须紧跟在脚本末尾，且上方不能有 %%)
% ========================================================================

function [metrics] = internal_master_blind_engine(fPath, mIdx, mName, fNames, bp_struct, res_raw)
    % 1. 文件夹初始化
    saveDir = [mName, '_Integrated_BlindTest_Results'];
    if ~exist(saveDir, 'dir'), mkdir(saveDir); end
    
    % 2. 载入并清洗数据 (强制前10列)
    data = readtable(fPath);
    X = table2array(data(:, 1:9)); Y = table2array(data(:, 10));
    v = ~all(X == 0, 2); X = X(v,:); Y = Y(v,:); numS = size(X,1);
    
    if isempty(bp_struct) || ~isfield(bp_struct, 'model')
        fprintf('   ⚠️ Error: Model data for [%s] is missing.\n', mName);
        metrics = zeros(1,5); return;
    end
    model = bp_struct.model;

    % 3. --- 核心分流：维度适配与量级强同步 (杜绝精度为负) ---
    try
        if mIdx <= 2 % PSO-SVR, FA-RF (自带标准化)
            Yp = predict(model, X);
        else % T3, T4, T5, T6 (强同步训练集比例尺)
            Xs = mapminmax('apply', X', bp_struct.ps_in);
            if mIdx == 4 % GA-BP
                Ypn = sim(model, Xs);
            elseif mIdx == 6 % LSTM
                Xc = cell(numS,1); for j=1:numS, Xc{j}=Xs(:,j); end
                Ypn = predict(model, Xc);
            else % PSO-LSBoost, PSO-LSSVM
                Ypn = predict(model, Xs')';
            end
            % [核心恢复] 将 0~1 的小数映射回真实 MPa 单位
            Yp = mapminmax('reverse', Ypn, bp_struct.ps_out)';
        end
    catch ME
        fprintf('   ❌ Model [%s] Failed: %s\n', mName, ME.message);
        metrics = zeros(1,5); return;
    end

    % 4. 深度审计统计 (1-RE 逻辑)
    Y = Y(:); Yp = Yp(:); Errs = Y - Yp; AbsE = abs(Errs);
    R2 = 1 - sum(Errs.^2)/sum((Y - mean(Y)).^2);
    RMSE = sqrt(mean(Errs.^2)); MAE = mean(AbsE);
    [max_err, ~] = max(AbsE); [min_err, ~] = min(AbsE);
    acc_v = (1 - (AbsE ./ (Y + eps))) * 100;
    metrics = [R2, RMSE, MAE, max_err, mean(acc_v)];

    % 5. 绘图 1: 盲测回归散点图 (包含最好/最差审计标注)
    h1 = figure('Name', [mName, '_Blind_Scatter'], 'Color', 'w', 'Position', [100 100 650 600]);
    scatter(Y, Yp, 85, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
    ref_l = [min([Y;Yp]) max([Y;Yp])]; plot(ref_l, ref_l, 'k--', 'LineWidth', 2);
    xlabel('Experimental Strength (MPa)', 'FontWeight', 'bold'); ylabel('Predicted Strength (MPa)', 'FontWeight', 'bold');
    
    st_box = {['Samples (N): ', num2str(numS)], ...
              ['Overall R^2: ', num2str(R2, '%.4f')], ...
              ['Mean MAE: ', num2str(MAE, '%.3f'), ' MPa'], ...
              ['Max Error: ', num2str(max_err, '%.2f'), ' MPa'], ...
              ['Best Acc: ', num2str(max(acc_v), '%.1f'), '%'], ...
              ['Worst Acc: ', num2str(min(acc_v), '%.1f'), '%']};
    annotation('textbox', [0.15, 0.6, 0.35, 0.35], 'String', st_box, 'FitBoxToText', 'on', 'BackgroundColor', 'w', 'FontWeight', 'bold');
    title([mName, ' Global Blind Audit']); grid on; axis square;
    save_all_formats_master(h1, saveDir, 'Blind_Regression_Scatter');

    % 6. 绘图 2: 盲测残差分布图 (带 ±MAE 辅助线)
    h2 = figure('Name', [mName, '_Blind_Residuals'], 'Color', 'w', 'Position', [150 150 700 500]);
    stem(Errs, 'filled', 'Color', [0.4 0.4 0.4]); hold on;
    yline(MAE, 'r--', 'LineWidth', 1.5); yline(-MAE, 'r--', 'LineWidth', 1.5);
    text(1, MAE+0.8, ['Accuracy Bound: \pm', num2str(MAE, '%.2f'), ' MPa'], 'Color', 'r', 'FontWeight', 'bold');
    xlabel('Blind Sample Index'); ylabel('Error (MPa)'); title('Residual Tolerance Analysis'); grid on;
    save_all_formats_master(h2, saveDir, 'Blind_Residual_Distribution');
    
    % 7. 绘图 3: SHAP 机制一致性分析图
    h3 = figure('Name', [mName, '_Blind_SHAP'], 'Color', 'w', 'Position', [200 200 800 600]);
    shap_v_blind = zeros(9, numS);
    for i = 1:numS
        curr_x = X(i,:); b_o = Yp(i);
        for f = 1:9
            tx = curr_x; tx(f) = mean(res_raw(:,f));
            if mIdx <= 2, yp_t = predict(model, tx);
            else, xt_s = mapminmax('apply', tx', bp_struct.ps_in);
                if mIdx == 4, yp_t = sim(model, xt_s); elseif mIdx == 6, yp_t = predict(model, {xt_s});
                else, yp_t = predict(model, xt_s')'; end
                yp_t = mapminmax('reverse', yp_t, bp_struct.ps_out);
            end
            shap_v_blind(f, i) = b_o - yp_t;
        end
    end
    for f_p = 1:9
        scatter(shap_v_blind(f_p, :), f_p + (rand(1, numS)-0.5)*0.3, 25, 'filled', 'MarkerFaceAlpha', 0.5); hold on;
    end
    set(gca, 'YTick', 1:9, 'YTickLabel', fNames); xlabel('SHAP Value (Impact on Model Output)'); 
    title(['SHAP Mechanism Audit: ', mName]); grid on;
    save_all_formats_master(h3, saveDir, 'Blind_SHAP_Analysis');
end

function auto_layout_manager_T7(fig_handle, main_t, ~)
    ax = findobj(fig_handle, 'Type', 'axes');
    for i = 1:length(ax)
        set(ax(i), 'Units', 'normalized'); p = get(ax(i), 'Position');
        set(ax(i), 'Position', [p(1), p(2)+0.06, p(3), p(4)*0.88]);
    end
    annotation(fig_handle, 'textbox', [0.05, 0.005, 0.9, 0.08], 'String', main_t, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontWeight', 'bold', 'FontSize', 11, 'Interpreter', 'none');
end

function save_all_formats_master(h, folder, name)
    saveas(h, fullfile(folder, [name, '.fig']));
    saveas(h, fullfile(folder, [name, '.png']));
    print(h, fullfile(folder, [name, '.svg']), '-dsvg', '-r600');
end
