%% ========================================================================
%  项目：ConcreteModel_LSBoost.mat 全维度科研 8 图套装 (彻底修复乱码)
%  修复：采用“图 8 式”显式注入法，确保全界面中文不产生方块
%% ========================================================================
clear; clc; close all;
file_name = 'ConcreteModel_LSBoost.mat';
if ~exist(file_name, 'file'); error('未找到数据文件'); end
load(file_name); 

% --- 核心字体配置 ---
if ispc; cFont = 'Microsoft YaHei'; else; cFont = 'SimSun'; end
eFont = 'Times New Roman';

% 颜色定义
c_blue = [0.15 0.35 0.65]; c_red = [0.85 0.33 0.1];

%% --- 图 1：测试集对比 ---
figure('Units', 'normalized', 'Position', [0.05 0.6 0.3 0.35], 'Name', 'Fig1');
plot(bp.T_te_real, '-s', 'Color', c_blue, 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'w'); hold on;
plot(bp.T_te_sim, '--o', 'Color', c_red, 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', 'w');
grid on; 
xlabel('Sample Index / 样本编号'); ylabel('Strength / 抗压强度 (MPa)');
lgd = legend({'Measured / 实验值', 'Predicted / 预测值'}, 'Location', 'best');
title({'Test Set Comparison','测试集性能对比'});
% 强制注入字体 (仿图8方法)
set(gca, 'FontName', cFont); 
set(findobj(gcf,'type','axes'),'FontName',cFont);
set(findobj(gcf,'type','text'),'FontName',cFont);

%% --- 图 2：测试集回归 ---
figure('Units', 'normalized', 'Position', [0.36 0.6 0.25 0.35], 'Name', 'Fig2');
scatter(bp.T_te_real, bp.T_te_sim, 50, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
ref_lim = [min([bp.T_te_real; bp.T_te_sim])-5, max([bp.T_te_real; bp.T_te_sim])+5];
plot(ref_lim, ref_lim, 'k--', 'LineWidth', 2);
grid on; axis square; 
xlabel('Measured / 实验值 (MPa)'); ylabel('Predicted / 预测值 (MPa)');
text(ref_lim(1)+5, ref_lim(2)-10, {['R^{2} = ', num2str(bp.R2, '%.4f')], ...
    ['MAE = ', num2str(bp.MAE, '%.3f')]}, 'FontWeight', 'bold');
title({'Regression Analysis (Test)','回归分析 (测试集)'});
set(gca, 'FontName', cFont); 
set(findobj(gcf,'Tag','Legend'),'FontName',cFont);

%% --- 图 3：训练集对比 ---
figure('Units', 'normalized', 'Position', [0.05 0.1 0.3 0.35], 'Name', 'Fig3');
plot(bp.T_tr_real, 'Color', [0.6 0.6 0.6]); hold on;
plot(bp.T_tr_sim, 'r.', 'MarkerSize', 7);
grid on; xlabel('Sample Index / 样本编号'); ylabel('Strength / 抗压强度 (MPa)');
legend({'Measured / 实验值', 'Predicted / 预测值'});
title({'Training Set Comparison','训练集性能对比'});
set(findobj(gcf,'type','axes'),'FontName',cFont);
set(findobj(gcf,'type','text'),'FontName',cFont);

%% --- 图 4：训练集回归 ---
figure('Units', 'normalized', 'Position', [0.36 0.1 0.25 0.35], 'Name', 'Fig4');
scatter(bp.T_tr_real, bp.T_tr_sim, 30, 'filled', 'MarkerFaceColor', [0.3 0.3 0.3]); hold on;
plot(ref_lim, ref_lim, 'r-', 'LineWidth', 1.5);
grid on; axis square; xlabel('Measured / 实验值 (MPa)'); ylabel('Predicted / 预测值 (MPa)');
title(['Train Regression (R^{2} = ', num2str(bp.R2_tr, '%.4f'), ')']);
set(gca, 'FontName', cFont); 

%% --- 图 5：相对误差分布 ---
figure('Units', 'normalized', 'Position', [0.62 0.6 0.3 0.35], 'Name', 'Fig5');
errors = (bp.T_te_sim - bp.T_te_real) ./ bp.T_te_real * 100;
histogram(errors, 'BinWidth', 2.5, 'FaceColor', c_blue, 'EdgeColor', 'w');
xlabel('Relative Error / 相对误差 (%)'); ylabel('Frequency / 频数');
grid on; title({'Error Distribution','相对误差分布'});
set(findobj(gcf,'type','axes'),'FontName',cFont);

%% --- 图 6：优化收敛曲线 ---
figure('Units', 'normalized', 'Position', [0.62 0.1 0.3 0.35], 'Name', 'Fig6');
plot(bp.conv, 'LineWidth', 2, 'Color', [0.1 0.5 0.2]);
grid on; xlabel('Iterations / 迭代次数'); ylabel('Fitness (MSE) / 适应度值');
title({'PSO Optimization Curve','PSO 优化收敛曲线'});
set(findobj(gcf,'type','axes'),'FontName',cFont);

%% --- 图 7：特征重要性 ---
[sorted_imp, idx] = sort(bp.importance, 'ascend');
figure('Units', 'normalized', 'Position', [0.05 0.35 0.4 0.35], 'Name', 'Fig7');
b = barh(sorted_imp/sum(sorted_imp)*100, 'FaceColor', 'flat');
b.CData = parula(9); 
set(gca, 'YTick', 1:9, 'YTickLabel', featureNames(idx), 'FontName', cFont); 
xlabel('Relative Importance / 相对重要性 (%)');
title({'Feature Importance Analysis','特征重要性分析'});
set(findobj(gcf,'type','text'),'FontName',cFont);

%% --- 图 8：相关性热力图 (基准参照) ---
figure('Units', 'normalized', 'Position', [0.5 0.35 0.4 0.45], 'Name', 'Fig8');
corr_mat = corr(res_raw);
imagesc(corr_mat); colormap(jet); colorbar; clim([-1 1]);
set(gca, 'XTick', 1:10, 'XTickLabel', [featureNames, 'Strength'], ...
         'YTick', 1:10, 'YTickLabel', [featureNames, 'Strength'], 'FontName', cFont);
xtickangle(45); axis square;
for i = 1:10; for j = 1:10
    text(j, i, num2str(corr_mat(i,j), '%.2f'), 'HorizontalAlignment', 'center', ...
        'FontSize', 8, 'FontName', eFont);
end; end
title({'Correlation Heatmap','特征相关性热力图'}, 'FontName', cFont);

fprintf('✅ 乱码修复完成！所有图表均强制注入了 %s 字体。\n', cFont);