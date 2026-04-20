function T3RC_V39()
% =====================================================================
% Project: Rubberized Concrete AI Design Platform (V39 Final Edition)
% Features: Centered UI Buttons, Auto-save FIG, ± Error Symbol
% =====================================================================
%% --- Module 1: Initialization ---
try
    baseDir = 'Research_Outputs';
    if ~exist(baseDir, 'dir'); mkdir(baseDir); end
    imgDir = fullfile(baseDir, 'Snapshots'); if ~exist(imgDir, 'dir'); mkdir(imgDir); end
    
    % --- 在 Module 1 加入以下文件夹逻辑 ---
    guiDir = fullfile(baseDir, 'GUI_Results'); % GUI主结果文件夹
    blindDir = fullfile(guiDir, 'Blind_Audit'); % 盲测子文件夹
    predDir = fullfile(guiDir, 'Batch_Prediction'); % 批量预测子文件夹
    if ~exist(guiDir, 'dir'); mkdir(guiDir); end
    if ~exist(blindDir, 'dir'); mkdir(blindDir); end
    if ~exist(predDir, 'dir'); mkdir(predDir); end
    CoreData.blindDir = blindDir; 
    CoreData.predDir = predDir;

    file_name = 'ConcreteModel_LSBoost.mat';
    vars = load(file_name);
    if isfield(vars, 'Best_Model'); CoreData.model = vars.Best_Model;
    else; CoreData.model = vars.bp.model; end
    
    CoreData.raw = vars.res_raw;
    CoreData.X_min = min(CoreData.raw(:, 1:9));
    CoreData.X_max = max(CoreData.raw(:, 1:9));
    CoreData.Y_min = min(CoreData.raw(:, 10));
    CoreData.Y_max = max(CoreData.raw(:, 10));
    CoreData.MAE = vars.bp.MAE;
    CoreData.imgDir = imgDir;
    CoreData.baseDir = baseDir;
catch ME
    errordlg(['Init Failed: ', ME.message]); return;
end
mainFont = 'Times New Roman';
persistent History_Y LogCells; 
History_Y = []; 
LogCells = cell(0, 11);

%% --- Module 2: GUI Design ---
fig = figure('Name', 'Rubberized Concrete AI Design Platform V39', ...
             'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.88], ...
             'Color', [1 1 1], 'MenuBar', 'none', 'NumberTitle', 'off');
fig.UserData = CoreData; 
fNamesEN = {'W/B', 'Rubber', 'Size', 'Cement', 'FineAgg', 'CoarseAgg', 'SF/C', 'SP', 'Age'};
colors = lines(9);

% --- Plotting Areas ---
axSHAP = axes(fig, 'Position', [0.05 0.55 0.22 0.35], 'Tag', 'axSHAP'); 
ax3D = axes(fig, 'Position', [0.32 0.55 0.32 0.35], 'Tag', 'ax3D');    
axHist = axes(fig, 'Position', [0.72 0.55 0.23 0.35], 'Tag', 'axHist');  

% --- Prediction Result Panel ---
pnlRes = uipanel(fig, 'Position', [0.35 0.38 0.3 0.07], 'BackgroundColor', [0.15 0.35 0.65], 'BorderType', 'none', 'Tag', 'pnlRes');
lblResult = uicontrol(pnlRes, 'Style', 'text', 'Units', 'normalized', 'Position', [0.02, 0.05, 0.96, 0.9], ...
                      'String', 'Ready', 'FontSize', 14, 'FontWeight', 'bold', 'ForegroundColor', 'w', ...
                      'BackgroundColor', [0.15 0.35 0.65], 'FontName', mainFont);

% --- Design Log Table ---
uicontrol(fig, 'Style', 'text', 'Units', 'normalized', 'Position', [0.65, 0.33, 0.33, 0.03], ...
          'String', 'Design Log', 'FontWeight', 'bold', 'FontName', mainFont, 'BackgroundColor', 'w');
colNames = {'ID', 'W/B', 'Rub', 'Size', 'Cem', 'Fine', 'Coarse', 'SF/C', 'SP', 'Age', 'Strength'};
hTable = uitable(fig, 'Units', 'normalized', 'Position', [0.65, 0.02, 0.33, 0.3], ...
                 'ColumnName', colNames, 'ColumnWidth', {30, 40, 40, 40, 40, 40, 45, 40, 40, 40, 55}, ...
                 'RowName', [], 'FontSize', 8, 'FontName', mainFont, 'Tag', 'hTable');

% --- Parameter Control Panel ---
pnlCtrl = uipanel(fig, 'Title', 'Parameter Control Console', 'Position', [0.02 0.02 0.62 0.32], ...
                  'BackgroundColor', [0.98 0.98 0.98], 'FontWeight', 'bold', 'FontSize', 10, 'FontName', mainFont, 'Tag', 'pnlCtrl');
hSliders = cell(9, 1); hEdits = cell(9, 1);
for i = 1:9
    row = ceil(i/3); col = mod(i-1, 3);
    x_base = 0.02 + col*0.33; y_base = 0.82 - (row-1)*0.36;
    uicontrol(pnlCtrl, 'Style', 'text', 'Units', 'normalized', 'Position', [x_base, y_base, 0.1, 0.16], ...
              'String', fNamesEN{i}, 'BackgroundColor', [0.98 0.98 0.98], ...
              'FontWeight', 'bold', 'FontName', mainFont, 'FontSize', 9);
    hSliders{i} = uicontrol(pnlCtrl, 'Style', 'slider', 'Units', 'normalized', 'Position', [x_base+0.1, y_base+0.02, 0.15, 0.1], ...
                           'Min', CoreData.X_min(i), 'Max', CoreData.X_max(i), 'Value', mean([CoreData.X_min(i), CoreData.X_max(i)]), ...
                           'Callback', @(~,~) update_forward(i, true), 'Tag', ['S', num2str(i)]);
    hEdits{i} = uicontrol(pnlCtrl, 'Style', 'edit', 'Units', 'normalized', 'Position', [x_base+0.26, y_base+0.02, 0.04, 0.15], ...
                         'String', num2str(hSliders{i}.Value, '%.2f'), ...
                         'Callback', @(src, ~) validate_input(src, i), 'Tag', ['E', num2str(i)]);
    uicontrol(pnlCtrl, 'Style', 'text', 'Units', 'normalized', 'Position', [x_base+0.05, y_base-0.12, 0.28, 0.1], ...
              'String', sprintf('Range: [%.1f - %.1f]', CoreData.X_min(i), CoreData.X_max(i)), ...
              'FontSize', 7, 'ForegroundColor', [0.5 0.5 0.5], 'BackgroundColor', [0.98 0.98 0.98], ...
              'HorizontalAlignment', 'left', 'FontName', mainFont);
end

% --- Functional Buttons (Centered Layout) ---
% --- 替换原来的 Functional Buttons 代码 (约第 93-117 行) ---此行为93行
% --- Module 2.5: Functional Buttons (3x2 Aesthetic Layout) ---
btn_w = 0.10; btn_h = 0.045; 
start_x = 0.66; 

% Row 1: Scientific Visualization & Snapshot
uicontrol(fig, 'Style', 'pushbutton', 'String', 'Pop FIG (Full)', 'Units', 'normalized', ...
          'Position', [start_x, 0.46, btn_w, btn_h], 'BackgroundColor', [0.4 0.2 0.6], ...
          'ForegroundColor', 'w', 'FontWeight', 'bold', 'Callback', @pop_fig_window, 'Tag', 'btn1');
uicontrol(fig, 'Style', 'pushbutton', 'String', 'Export Snapshot', 'Units', 'normalized', ...
          'Position', [start_x + btn_w + 0.01, 0.46, btn_w, btn_h], 'BackgroundColor', [0.2 0.5 0.3], ...
          'ForegroundColor', 'w', 'FontWeight', 'bold', 'Callback', @export_gui, 'Tag', 'btn2');
uicontrol(fig, 'Style', 'pushbutton', 'String', 'Blind Test Audit', 'Units', 'normalized', ...
          'Position', [start_x + 2*(btn_w + 0.01), 0.46, btn_w, btn_h], 'BackgroundColor', [0.8 0.4 0.1], ...
          'ForegroundColor', 'w', 'FontWeight', 'bold', 'Callback', @gui_blind_test_engine, 'Tag', 'btn3');

% Row 2: History & Batch Processing
uicontrol(fig, 'Style', 'pushbutton', 'String', 'Replay (15 Steps)', 'Units', 'normalized', ...
          'Position', [start_x, 0.40, btn_w, btn_h], 'BackgroundColor', [0.6 0.3 0.1], ...
          'ForegroundColor', 'w', 'FontWeight', 'bold', 'Callback', @replay_recent, 'Tag', 'btn4');
uicontrol(fig, 'Style', 'pushbutton', 'String', 'Export Excel Log', 'Units', 'normalized', ...
          'Position', [start_x + btn_w + 0.01, 0.40, btn_w, btn_h], 'BackgroundColor', [0.1 0.4 0.6], ...
          'ForegroundColor', 'w', 'FontWeight', 'bold', 'Callback', @export_log_excel, 'Tag', 'btn5');
uicontrol(fig, 'Style', 'pushbutton', 'String', 'Batch Prediction', 'Units', 'normalized', ...
          'Position', [start_x + 2*(btn_w + 0.01), 0.40, btn_w, btn_h], 'BackgroundColor', [0.1 0.6 0.8], ...
          'ForegroundColor', 'w', 'FontWeight', 'bold', 'Callback', @gui_batch_predict_engine, 'Tag', 'btn6');

update_forward(1, true);

%% --- Logic Functions ---
    function update_forward(idx, isManual)
        for k = 1:9
            val = hSliders{k}.Value;
            if k == 9, val = round(val); hSliders{k}.Value = val; end 
            hEdits{k}.String = num2str(val, '%.2f');
        end
        X_curr = cellfun(@(h) str2double(h.String), hEdits)';
        px = (X_curr - CoreData.X_min) ./ (CoreData.X_max - CoreData.X_min + eps);
        tx = predict(CoreData.model, px);
        Y_curr = tx * (CoreData.Y_max - CoreData.Y_min) + CoreData.Y_min;
        
        set(lblResult, 'String', sprintf('Predicted Strength: %.2f MPa\nError Range: ±%.2f MPa', Y_curr, CoreData.MAE));
        History_Y = [History_Y, Y_curr]; if length(History_Y) > 30, History_Y(1) = []; end
        
        if isManual
            newRow = [ {size(LogCells,1)+1}, num2cell(X_curr), {sprintf('%.2f', Y_curr)} ];
            LogCells = [newRow; LogCells];
            if size(LogCells,1) > 50, LogCells(51:end,:) = []; end
            hTable.Data = LogCells;
        end
        draw_SHAP(X_curr); draw_Star(X_curr, Y_curr); draw_Trace();
    end

    function validate_input(src, i)
        val = str2double(src.String);
        if isnan(val) || val < CoreData.X_min(i) || val > CoreData.X_max(i)
            msgbox(sprintf('Out of range: [%.1f - %.1f]', CoreData.X_min(i), CoreData.X_max(i)), 'Error');
            src.String = num2str(hSliders{i}.Value, '%.2f'); 
        else
            if i == 9, val = round(val); src.String = num2str(val); end 
            hSliders{i}.Value = val; update_forward(i, true);
        end
    end

    function draw_SHAP(X)
        axes(axSHAP); cla; hold on;
        mu = mean(CoreData.raw(:, 1:9));
        diffs = (X - mu) ./ (CoreData.X_max - CoreData.X_min + eps);
        for i = 1:9
            c = [0.2 0.5 0.7]; if diffs(i) < 0, c = [0.8 0.3 0.2]; end
            barh(i, diffs(i), 'FaceColor', c, 'EdgeColor', 'none');
            text(diffs(i), i, sprintf(' %.2f', diffs(i)), 'FontSize', 8, 'FontName', mainFont);
        end
        set(axSHAP, 'YTick', 1:9, 'YTickLabel', fNamesEN, 'YDir', 'reverse', 'FontName', mainFont);
        xlabel('Contribution', 'FontName', mainFont);
        title('Fig. (a) SHAP Analysis', 'FontName', mainFont);
    end

    function draw_Star(X, Y)
        axes(ax3D); cla; hold on;
        theta = linspace(0, 2*pi, 10); theta(end) = []; xr = 12*cos(theta); yr = 12*sin(theta);
        for i = 1:9
            h = (X(i) - CoreData.X_min(i)) / (CoreData.X_max(i) - CoreData.X_min(i) + eps) * 45;
            [bx, by, bz] = cylinder(0.6, 20);
            surf(bx + xr(i), by + yr(i), bz * h, 'FaceColor', colors(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.6);
            text(xr(i)*1.3, yr(i)*1.3, 0, fNamesEN{i}, 'HorizontalAlignment', 'center', 'FontSize', 8, 'FontName', mainFont);
            line([xr(i), 0], [yr(i), 0], [h, Y], 'Color', [0.8 0.8 0.8], 'LineStyle', ':');
        end
        plot3(0, 0, Y, 'kp', 'MarkerSize', 18, 'MarkerFaceColor', [1 0.8 0]);
        text(0, 0, Y+8, sprintf('%.1f MPa', Y), 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'FontName', mainFont);
        view(35, 25); grid on; zlim([0 100]); set(gca, 'XTick', [], 'YTick', [], 'FontName', mainFont);
        title('Fig. (b) Mechanism Monitor', 'FontName', mainFont);
    end

    function draw_Trace()
        axes(axHist); cla; hold on;
        if length(History_Y) > 1
            fill([1:length(History_Y), fliplr(1:length(History_Y))], ...
                 [History_Y-CoreData.MAE, fliplr(History_Y+CoreData.MAE)], [0.85 0.9 1], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
        end
        plot(History_Y, '-s', 'LineWidth', 1.5, 'Color', [0.15 0.35 0.65], 'MarkerFaceColor', 'w');
        grid on; ylabel('Strength (MPa)', 'FontName', mainFont); xlabel('Adjustment Steps', 'FontName', mainFont);
        title('Fig. (c) Sensitivity Trace', 'FontName', mainFont);
    end

    function replay_recent(~, ~)
        if size(LogCells, 1) < 2, return; end
        numToReplay = min(size(LogCells, 1), 15);
        for s = numToReplay:-1:1
            row_data = cell2mat(LogCells(s, 2:10));
            for k = 1:9, hSliders{k}.Value = row_data(k); end
            update_forward(1, false); 
            drawnow; pause(0.3);
        end
    end

   function pop_fig_window(~, ~)
        popFig = figure('Name', 'Full UI Scientific Export', 'Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
        % 动态搜索并克隆所有带有 btn 标签的按钮
        allBtns = [];
        for b_i = 1:6
            allBtns = [allBtns, findobj(fig, 'Tag', ['btn', num2str(b_i)])];
        end
        copyobj([axSHAP, ax3D, axHist, pnlRes, hTable, pnlCtrl, allBtns], popFig);
        ts = datestr(now, 'yyyy-mm-dd_HHMMSS');
        fname = fullfile(CoreData.baseDir, ['Scientific_Figure_', ts, '.fig']);
        saveas(popFig, fname);
        msgbox(['Full UI saved with 6 buttons to: ', fname], 'Success');
    end

    function export_log_excel(~, ~)
        if isempty(LogCells), return; end
        T = cell2table(LogCells, 'VariableNames', colNames);
        fname = fullfile(CoreData.baseDir, ['Design_Log_', datestr(now, 'yyyy-mm-dd_HHMMSS'), '.xlsx']);
        writetable(T, fname); msgbox(['Saved to: ', fname], 'Success');
    end

    function export_gui(~, ~)
        fname = fullfile(CoreData.imgDir, ['Snapshot_', datestr(now, 'yyyy-mm-dd_HHMMSS'), '.png']);
        exportgraphics(fig, fname, 'Resolution', 300); msgbox(['Saved to: ', fname], 'Success');
        %% ========================== 新增：盲测审计引擎 ==========================
% --- 确保贴在 export_gui 的 end 之后 ---
    
    function gui_blind_test_engine(~, ~)
        [file, path] = uigetfile('*.xlsx', 'Select Blind Test Data');
        if isequal(file, 0); return; end
        data = readmatrix(fullfile(path, file));
        X_raw = data(:, 1:9); Y_act = data(:, 10);
        px = (X_raw - CoreData.X_min) ./ (CoreData.X_max - CoreData.X_min + eps);
        py = predict(CoreData.model, px);
        Y_pre = py * (CoreData.Y_max - CoreData.Y_min) + CoreData.Y_min;
        plot_scientific_audit(Y_act, Y_pre, X_raw, CoreData.blindDir, 'Blind_Audit');
    end

    function gui_batch_predict_engine(~, ~)
        [file, path] = uigetfile('*.xlsx', 'Select Batch Prediction Data');
        if isequal(file, 0); return; end
        opts = questdlg('Does the last column contain Actual Strength?', 'Data Check', 'Yes', 'No', 'No');
        data = readmatrix(fullfile(path, file));
        if strcmp(opts, 'Yes'), X_raw = data(:, 1:9); Y_act = data(:, 10);
        else, X_raw = data(:, 1:9); Y_act = []; end
        px = (X_raw - CoreData.X_min) ./ (CoreData.X_max - CoreData.X_min + eps);
        py = predict(CoreData.model, px);
        Y_pre = py * (CoreData.Y_max - CoreData.Y_min) + CoreData.Y_min;
        T = array2table(X_raw, 'VariableNames', fNamesEN);
        if ~isempty(Y_act); T.Actual_Strength = Y_act; end
        T.Predicted_Strength = Y_pre;
        outName = fullfile(CoreData.predDir, ['Batch_Result_', datestr(now, 'HHMMSS'), '.xlsx']);
        writetable(T, outName);
        if strcmp(opts, 'Yes'), plot_scientific_audit(Y_act, Y_pre, X_raw, CoreData.predDir, 'Batch_Predict'); end
        msgbox(['Success! Excel saved to: ', outName], 'Success');
    end

    function plot_scientific_audit(Ya, Yp, Xr, folder, tag)
        % 1. 散点图 (标注 R2, MAE, RMSE)
        h1 = figure('Color', 'w', 'Name', [tag, '_Scatter']);
        scatter(Ya, Yp, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
        plot([min(Ya) max(Ya)], [min(Ya) max(Ya)], 'k--', 'LineWidth', 2);
        r2 = 1 - sum((Ya-Yp).^2)/sum((Ya-mean(Ya)).^2);
        rmse = sqrt(mean((Ya-Yp).^2));
        title(sprintf('R^2: %.4f | RMSE: %.2f | MAE: %.2f', r2, rmse, mean(abs(Ya-Yp))));
        xlabel('Experimental (MPa)'); ylabel('Predicted (MPa)'); grid on;
        saveas(h1, fullfile(folder, [tag, '_Scatter_', datestr(now, 'HHMMSS'), '.fig']));
        
        % 2. 残差分布图
        h2 = figure('Color', 'w', 'Name', [tag, '_Residual']);
        stem(Ya-Yp, 'filled'); grid on; title('Residual Tolerance Audit');
        ylabel('Error (MPa)'); xlabel('Sample Index');
        saveas(h2, fullfile(folder, [tag, '_Residual_', datestr(now, 'HHMMSS'), '.fig']));
        
        % 3. SHAP 简易对比图
        h3 = figure('Color', 'w', 'Name', [tag, '_SHAP']);
        mu = mean(CoreData.raw(:, 1:9));
        shap_proxy = (Xr - mu) ./ (CoreData.X_max - CoreData.X_min + eps);
        barh(mean(abs(shap_proxy))); set(gca, 'YTick', 1:9, 'YTickLabel', fNamesEN);
        title('Mechanism Consistency Analysis'); xlabel('Impact Strength'); grid on;
        saveas(h3, fullfile(folder, [tag, '_SHAP_', datestr(now, 'HHMMSS'), '.fig']));
    end

% --- 这是整个 T3RC_V39.m 的最后一个 end ---
    end
% --- 功能模块：盲测与批量预测 (注入主函数作用域) ---
    
    function gui_blind_test_engine(~, ~)
        [file, path] = uigetfile('*.xlsx', 'Select Blind Test Data');
        if isequal(file, 0); return; end
        data = readmatrix(fullfile(path, file));
        X_raw = data(:, 1:9); Y_act = data(:, 10);
        % 归一化与预测
        px = (X_raw - CoreData.X_min) ./ (CoreData.X_max - CoreData.X_min + eps);
        py = predict(CoreData.model, px);
        Y_pre = py * (CoreData.Y_max - CoreData.Y_min) + CoreData.Y_min;
        plot_scientific_audit(Y_act, Y_pre, X_raw, CoreData.blindDir, 'Blind_Audit');
    end

    function gui_batch_predict_engine(~, ~)
        [file, path] = uigetfile('*.xlsx', 'Select Batch Prediction Data');
        if isequal(file, 0); return; end
        opts = questdlg('Does the last column contain Actual Strength?', 'Data Check', 'Yes', 'No', 'No');
        data = readmatrix(fullfile(path, file));
        if strcmp(opts, 'Yes'), X_raw = data(:, 1:9); Y_act = data(:, 10);
        else, X_raw = data(:, 1:9); Y_act = []; end
        % 预测流
        px = (X_raw - CoreData.X_min) ./ (CoreData.X_max - CoreData.X_min + eps);
        py = predict(CoreData.model, px);
        Y_pre = py * (CoreData.Y_max - CoreData.Y_min) + CoreData.Y_min;
        % 导出表格
        T = array2table(X_raw, 'VariableNames', fNamesEN);
        if ~isempty(Y_act); T.Actual_Strength = Y_act; end
        T.Predicted_Strength = Y_pre;
        outName = fullfile(CoreData.predDir, ['Batch_Result_', datestr(now, 'HHMMSS'), '.xlsx']);
        writetable(T, outName);
        if strcmp(opts, 'Yes'), plot_scientific_audit(Y_act, Y_pre, X_raw, CoreData.predDir, 'Batch_Predict'); end
        msgbox(['Batch Process Success! Saved to: ', outName], 'Success');
    end

    function plot_scientific_audit(Ya, Yp, Xr, folder, tag)
        % 1. 散点图 (标注 R2, RMSE, MAE)
        h1 = figure('Color', 'w', 'Name', [tag, '_Scatter']);
        scatter(Ya, Yp, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
        plot([min(Ya) max(Ya)], [min(Ya) max(Ya)], 'k--', 'LineWidth', 2);
        r2 = 1 - sum((Ya-Yp).^2)/sum((Ya-mean(Ya)).^2);
        rmse = sqrt(mean((Ya-Yp).^2));
        title(sprintf('R^2: %.4f | RMSE: %.2f | MAE: %.2f', r2, rmse, mean(abs(Ya-Yp))));
        xlabel('Experimental (MPa)'); ylabel('Predicted (MPa)'); grid on;
        saveas(h1, fullfile(folder, [tag, '_Scatter_', datestr(now, 'HHMMSS'), '.fig']));
        
        % 2. 残差分布图
        h2 = figure('Color', 'w', 'Name', [tag, '_Residual']);
        stem(Ya-Yp, 'filled'); grid on; title('Residual Tolerance Audit');
        ylabel('Error (MPa)'); xlabel('Sample Index');
        saveas(h2, fullfile(folder, [tag, '_Residual_', datestr(now, 'HHMMSS'), '.fig']));
        
        % 3. SHAP 简易对比图
        h3 = figure('Color', 'w', 'Name', [tag, '_SHAP']);
        mu = mean(CoreData.raw(:, 1:9));
        shap_proxy = (Xr - mu) ./ (CoreData.X_max - CoreData.X_min + eps);
        barh(mean(abs(shap_proxy))); set(gca, 'YTick', 1:9, 'YTickLabel', fNamesEN);
        title('Mechanism Consistency Analysis'); xlabel('Impact Strength'); grid on;
        saveas(h3, fullfile(folder, [tag, '_SHAP_', datestr(now, 'HHMMSS'), '.fig']));
    end
end