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
   % --- 修改后的逻辑 ---
    uicontrol(pnlCtrl, 'Style', 'text', 'Units', 'normalized', 'Position', [x_base+0.05, y_base-0.12, 0.28, 0.1], ...
              'String', sprintf('Range: [%.1f - %.1f]', CoreData.X_min(i), CoreData.X_max(i)), ...
              'FontSize', 9, ... % 调大字号
              'FontWeight', 'bold', ... % 加粗
              'ForegroundColor', [0 0.3 0.6], ... % 改为较亮的深蓝色（亮且清晰）
              'BackgroundColor', [0.98 0.98 0.98], ...
              'HorizontalAlignment', 'left', 'FontName', mainFont);
end

% --- Functional Buttons (Centered Layout) ---
% --- 替换原来的 Functional Buttons 代码 (约第 93-117 行) ---此行为93行
% --- Module 2.5: Functional Buttons (3x2 Aesthetic Layout) ---
btn_w = 0.10; btn_h = 0.045; 
start_x = 0.66; 

% Row 1: Scientific Visualization & Snapshot (Y从0.46改为0.41)
uicontrol(fig, 'Style', 'pushbutton', 'String', 'Pop FIG (Full)', 'Units', 'normalized', ...
          'Position', [start_x, 0.41, btn_w, btn_h], 'BackgroundColor', [0.4 0.2 0.6], ...
          'ForegroundColor', 'w', 'FontWeight', 'bold', 'Callback', @pop_fig_window, 'Tag', 'btn1');
uicontrol(fig, 'Style', 'pushbutton', 'String', 'Export Snapshot', 'Units', 'normalized', ...
          'Position', [start_x + btn_w + 0.01, 0.41, btn_w, btn_h], 'BackgroundColor', [0.2 0.5 0.3], ...
          'ForegroundColor', 'w', 'FontWeight', 'bold', 'Callback', @export_gui, 'Tag', 'btn2');
uicontrol(fig, 'Style', 'pushbutton', 'String', 'Blind Test Audit', 'Units', 'normalized', ...
          'Position', [start_x + 2*(btn_w + 0.01), 0.41, btn_w, btn_h], 'BackgroundColor', [0.8 0.4 0.1], ...
          'ForegroundColor', 'w', 'FontWeight', 'bold', 'Callback', @gui_blind_test_engine, 'Tag', 'btn3');

% Row 2: History & Batch Processing (Y从0.40改为0.36)
uicontrol(fig, 'Style', 'pushbutton', 'String', 'Replay (15 Steps)', 'Units', 'normalized', ...
          'Position', [start_x, 0.36, btn_w, btn_h], 'BackgroundColor', [0.6 0.3 0.1], ...
          'ForegroundColor', 'w', 'FontWeight', 'bold', 'Callback', @replay_recent, 'Tag', 'btn4');
uicontrol(fig, 'Style', 'pushbutton', 'String', 'Export Excel Log', 'Units', 'normalized', ...
          'Position', [start_x + btn_w + 0.01, 0.36, btn_w, btn_h], 'BackgroundColor', [0.1 0.4 0.6], ...
          'ForegroundColor', 'w', 'FontWeight', 'bold', 'Callback', @export_log_excel, 'Tag', 'btn5');
uicontrol(fig, 'Style', 'pushbutton', 'String', 'Batch Prediction', 'Units', 'normalized', ...
          'Position', [start_x + 2*(btn_w + 0.01), 0.36, btn_w, btn_h], 'BackgroundColor', [0.1 0.6 0.8], ...
          'ForegroundColor', 'w', 'FontWeight', 'bold', 'Callback', @gui_batch_predict_engine, 'Tag', 'btn6');


      % --- Module 2.6: 新控制变量法模块按钮 (放置在图a下方) ---
    % 修正位置：将其放置在图a(axSHAP)正下方的空白区域，y轴坐标降至0.4
    uicontrol(fig, 'Style', 'pushbutton', 'String', 'VCT Paradigm (Modified Controlled Variable Method)', ...
             'Units', 'normalized', 'Position', [0.035, 0.4, 0.25, 0.05], ...%这里不要变，要一直是左右0.035，高低0.4，长短0.25，宽细0.05
              'BackgroundColor', [0.1, 0.7, 0.4], 'ForegroundColor', 'w', ...
              'FontWeight', 'bold', 'FontSize', 10, ...
              'Callback', @open_vct_module, 'Tag', 'btnVCT');
  
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
            if size(LogCells,1) > 50, LogCells(350:end,:) = []; end
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
       
        % 确定 Y=13.0 为固定初始位置，处于 Contribution 标签下方
        text(0, 12.0, 'Fig. (a) SHAP Analysis', 'FontName', mainFont, ...
             'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end

   % --- 修改后的逻辑 ---
    function draw_Star(X, Y)
        axes(ax3D); cla; hold on;
        
        % 定义 9 个柱子的基准圆周位置 (半径=12)
        theta = linspace(0, 2*pi, 10); theta(end) = []; 
        xr = 12 * cos(theta); 
        yr = 12 * sin(theta); 
        
        for i = 1:9
            % 1. 绘制柱子 (Center: [xr(i), yr(i)], Base Z: 0)
            h = (X(i) - CoreData.X_min(i)) / (CoreData.X_max(i) - CoreData.X_min(i) + eps) * 45;
            [bx, by, bz] = cylinder(0.6, 20);
            surf(bx + xr(i), by + yr(i), bz * h, 'FaceColor', colors(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.6);
            
            % --- 修改后的逻辑 ---
            % 2. 定义变量名称的精确 (x, y, z) 坐标
            posX = xr(i) * 1.0; 
            posY = yr(i) * 1.0;
            
            % 3. 动态调整 Z 轴位置与对齐方式
            targetBelow = {'SF/C', 'SP', 'Age', 'W/B', 'Rubber'};
            targetAbove = {'Size', 'Cement', 'FineAgg', 'CoarseAgg'};
            
            if ismember(fNamesEN{i}, targetBelow)
                posZ = 0;          % 固定在地面
                vAlign = 'top';    % 位于点(0)的下方
            elseif ismember(fNamesEN{i}, targetAbove)
                posZ = h + 5;      % 随柱子高度 h 动态变动，始终领先 5 个单位
                vAlign = 'bottom'; % 位于点(h+5)的上方
            else
                posZ = 0; 
                vAlign = 'middle';
            end
            
            % 4. 绘制有色标题
            text(posX, posY, posZ, fNamesEN{i}, ...
                 'HorizontalAlignment', 'center', ...
                 'VerticalAlignment', vAlign, ...
                 'FontSize', 8, 'FontName', mainFont, ...
                 'FontWeight', 'bold', ...
                 'Color', colors(i,:)); 
                 
            % 绘制连接中心预测点的机构连线
            line([xr(i), 0], [yr(i), 0], [h, Y], 'Color', [0.8 0.8 0.8], 'LineStyle', ':');
        end
        
        % 5. 绘制中心预测星 (Coordinate: [0, 0, Y])
        plot3(0, 0, Y, 'kp', 'MarkerSize', 18, 'MarkerFaceColor', [1 0.8 0]);
        text(0, 0, Y + 8, sprintf('%.1f MPa', Y), 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'FontName', mainFont);
        
        % 6. 图表全局设置与主标题 (x, y, z) 定位
        view(35, 25); grid on; zlim([0 100]); 
        set(gca, 'XTick', [], 'YTick', [], 'FontName', mainFont);
        
        % 主标题位置坐标 [15, -25, -15]，确保在 3D 旋转默认视角下处于最下方
        text(16, -20, -10, 'Fig. (b) Mechanism Monitor', 'FontName', mainFont, ...
             'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end

    function draw_Trace()
        axes(axHist); cla; hold on;
        if length(History_Y) > 1
            fill([1:length(History_Y), fliplr(1:length(History_Y))], ...
                 [History_Y-CoreData.MAE, fliplr(History_Y+CoreData.MAE)], [0.85 0.9 1], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
        end
        plot(History_Y, '-s', 'LineWidth', 1.5, 'Color', [0.15 0.35 0.65], 'MarkerFaceColor', 'w');
        % --- 修改后的逻辑 ---
        grid on; ylabel('Strength (MPa)', 'FontName', mainFont); 
        xlabel('Adjustment Steps', 'FontName', mainFont);
        
        % 取消顶部 title，改用 text 置于 X 轴下方
        % 使用归一化单位 (Normalized) 确保位置固定在坐标轴正下方
        text(0.5, -0.2, 'Fig. (c) Sensitivity Trace', 'Units', 'normalized', ...
             'FontName', mainFont, 'FontSize', 10, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center');
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
        
        % 1. 获取常规按钮列表
        allBtns = [];
        for b_i = 1:6
            allBtns = [allBtns, findobj(fig, 'Tag', ['btn', num2str(b_i)])];
        end
        
        % 2. 获取绿色 VCT 按钮对象
        btnVCT_old = findobj(fig, 'Tag', 'btnVCT');
        
        % 3. 执行克隆
        % 先克隆大组件
        copyobj([axSHAP, ax3D, axHist, pnlRes, hTable, pnlCtrl, allBtns], popFig);
        
        % 4. 独立克隆并修正 VCT 按钮位置（防止显示不全）
        btnVCT_new = copyobj(btnVCT_old, popFig);
        % 强制修正位置：保持原比例但确保其在 popFig 的可见范围内
        set(btnVCT_new, 'Units', 'normalized', 'Position', [0.035, 0.4, 0.29, 0.05]); %长短是第三个数值
        
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
%%-------------------

% --- 这是整个的最后一个 end ---
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
%% --- Module 6: 新控制变量法 (VCT) 深度优化版 (V40+ 增强版) ---
    function open_vct_module(~, ~)
        % 1. 初始化窗口
        vctFig = figure('Name', 'VCT Paradigm - 虚拟受控实验四层验证体系', 'Units', 'normalized', ...
                        'Position', [0.03, 0.05, 0.94, 0.88], 'Color', 'w', 'MenuBar', 'none');
        
        % 核心修改：定义全局文件计数器，用于实现 P 编号全局递增
        vctFileCounter = zeros(1, 4); 

       % % 在界面最上方加入：批量预测、一键修改标签、全界面导出按钮
       % --- 调整后的居中按钮布局 ---
        btnW = 0.20; % 按钮宽度
        uicontrol(vctFig, 'Style', 'pushbutton', 'String', '▶ Batch Prediction', 'Units', 'normalized', ...
                  'Position', [0.19, 0.94, btnW, 0.04], 'BackgroundColor', [0.85, 0.33, 0.1], 'ForegroundColor', 'w', ...
                  'FontSize', 10, 'FontWeight', 'bold', 'Callback', @vct_batch_predict_wrapper);

        uicontrol(vctFig, 'Style', 'pushbutton', 'String', '✎ Update All Labels', 'Units', 'normalized', ...
                  'Position', [0.40, 0.94, btnW, 0.04], 'BackgroundColor', [0.2, 0.6, 0.5], 'ForegroundColor', 'w', ...
                  'FontSize', 10, 'FontWeight', 'bold', 'Callback', @update_vct_labels);

        uicontrol(vctFig, 'Style', 'pushbutton', 'String', '💾 Export Full FIG', 'Units', 'normalized', ...
                  'Position', [0.61, 0.94, btnW, 0.04], 'BackgroundColor', [0.4, 0.2, 0.6], 'ForegroundColor', 'w', ...
                  'FontSize', 10, 'FontWeight', 'bold', 'Callback', @export_full_ui);

        % --- 修正后的标签更新函数 ---
        function update_vct_labels(~, ~)
            prompt = {'Enter X-Axis Label:', 'Enter Y-Axis Label:'};
            definput = {'Replacement Ratio (%)', 'Compressive Strength (MPa)'};
            answer = inputdlg(prompt, 'Batch Update Labels', [1 50], definput);
            if ~isempty(answer)
                newX = answer{1}; newY = answer{2};
                % 关键：通过 vctFig 句柄直接查找坐标轴对象
                allAxes = findall(vctFig, 'Type', 'axes');
                for i = 1:length(allAxes)
                    xlabel(allAxes(i), newX);
                    ylabel(allAxes(i), newY);
                end
            end
        end

      
        % 嵌套函数：实现自动保存+弹出界面调整格式
        function export_full_ui(~,~)
            % 1. 自动保存当前状态
            defaultName = fullfile('Research_Outputs', ['VCT_Full_UI_', datestr(now, 'HHMMSS')]);
            if ~exist('Research_Outputs', 'dir'); mkdir('Research_Outputs'); end
            savefig(vctFig, [defaultName, '.fig']);
            
            % 2. 弹出整个新界面供用户调整（克隆模式）
            editFig = figure('Name', 'VCT Layout Editor (Editable)', 'Color', 'w', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.9]);
            copyobj(allchild(vctFig), editFig); 
            
            % 3. 弹出另存为对话框
            [file, path] = uiputfile({'*.fig','MATLAB Figure (*.fig)'; '*.png','PNG Image (*.png)'; '*.pdf','PDF Document (*.pdf)'}, ...
                                     'Save/Export VCT', defaultName);
            if ischar(file)
                saveas(vctFig, fullfile(path, file));
                msgbox(['Successfully exported to: ', file], 'Success');
            end
        end

        titles = {'1. Physical Consistency Boundary Verification', '2. Full-Range Statistical Feature Anchor Analysis', ...
                  '3. Output-Level Gradient Response Analysis', '4. Engineering Conventional Interval Focus Analysis'};
        pos_axes = {[0.06, 0.54, 0.35, 0.33], [0.55, 0.54, 0.35, 0.33], ...
                    [0.06, 0.08, 0.35, 0.33], [0.55, 0.08, 0.35, 0.33]};
        
        for k = 1:4
            ax = axes(vctFig, 'Position', pos_axes{k}, 'Tag', ['ax', num2str(k)]);
            title(ax, titles{k}, 'FontSize', 12, 'FontWeight', 'bold');
            grid(ax, 'on'); hold(ax, 'on'); box(ax, 'on');
            
            % 交互组件配置
            px = pos_axes{k}(1) + pos_axes{k}(3) + 0.005; py = pos_axes{k}(2) + 0.05;%%左右0.035，高低0.4，长短0.25，宽细0.02
            uicontrol(vctFig, 'Style', 'text', 'String', 'X-Axis:', 'Units', 'normalized', 'Position', [px, py+0.22, 0.03, 0.02], 'BackgroundColor', 'w');
            uicontrol(vctFig, 'Style', 'edit', 'String', '4', 'Units', 'normalized', 'Position', [px+0.028, py+0.22, 0.03, 0.03], 'Tag', ['editX', num2str(k)]); 
            uicontrol(vctFig, 'Style', 'text', 'String', 'Y-Axis:', 'Units', 'normalized', 'Position', [px, py+0.18, 0.03, 0.02], 'BackgroundColor', 'w');
            uicontrol(vctFig, 'Style', 'edit', 'String', '-1', 'Units', 'normalized', 'Position', [px+0.028, py+0.18, 0.03, 0.03], 'Tag', ['editY', num2str(k)]); 
            
            uicontrol(vctFig, 'Style', 'pushbutton', 'String', 'Import Excel', 'Units', 'normalized', ...
                      'Position', [px, py+0.10, 0.06, 0.04], 'BackgroundColor', [0.15, 0.35, 0.65], 'ForegroundColor', 'w', ...
                      'Callback', @(s,e) run_vct_sub_analysis(k));
            uicontrol(vctFig, 'Style', 'pushbutton', 'String', 'Pop FIG (Full)', 'Units', 'normalized', ...
                      'Position', [px, py+0.05, 0.06, 0.035], 'BackgroundColor', [0.4, 0.4, 0.4], 'ForegroundColor', 'w', ...
                      'Callback', @(s,e) save_vct_subplot(k));
        end

        % 手动保存并弹出 FIG 格式图窗口
        function save_vct_subplot(idx)
            subFig = figure('Name', ['子图', num2str(idx), ' 导出预览'], 'Color', 'w'); 
            oldAx = findobj(vctFig, 'Tag', ['ax', num2str(idx)]);
            newAx = copyobj(oldAx, subFig);
            set(newAx, 'Position', [0.15, 0.15, 0.7, 0.7]); 
            outP = fullfile('Research_Outputs', 'VCT_Results', 'SubPlots');
            if ~exist(outP, 'dir'); mkdir(outP); end
            save_name = fullfile(outP, ['子图', num2str(idx), '_', datestr(now, 'HHMMSS'), '.fig']);
            savefig(subFig, save_name);
            % 保存后不关闭窗口，保留界面供操作
            msgbox(['子图', num2str(idx), ' 已保存并弹出预览。'], 'Save Success');
        end

               function vct_batch_predict_wrapper(~, ~)
           % 1. 环境准备：创建 VCT 专用预测目录
          vctPredDir = fullfile('Research_Outputs', 'VCT_Results', 'Batch_Prediction');
            if ~exist(vctPredDir, 'dir'); mkdir(vctPredDir); end
            
            % 2. 交互环节：弹出文件夹/文件选择框
            [file, pth] = uigetfile('*.xlsx', 'Select Excel Data for VCT Batch Prediction');
            if isequal(file, 0); return; end % 用户取消则退出
            
            % 3. 执行预测：读取数据并利用核心模型计算
            data = readmatrix(fullfile(pth, file));
            X_raw = data(:, 1:9); % 提取前9列特征
            px = (X_raw - CoreData.X_min) ./ (CoreData.X_max - CoreData.X_min + eps); % 归一化
            py = predict(CoreData.model, px); % 虚拟推演
            Y_pre = py * (CoreData.Y_max - CoreData.Y_min) + CoreData.Y_min; % 反归一化
            
            % 4. 结果保存：构建表格并保存至 VCT 指定文件夹
            T = array2table([X_raw, Y_pre], 'VariableNames', [fNamesEN, {'Predicted_Strength'}]);
            outName = fullfile(vctPredDir, ['VCT_Batch_Result_', datestr(now, 'HHMMSS'), '.xlsx']);
            writetable(T, outName);
            
            msgbox(['VCT Batch Prediction Success! File saved to: ', outName], 'Success');
        end

        function run_vct_sub_analysis(idx)
            curAx = findobj(vctFig, 'Tag', ['ax', num2str(idx)]);
            eX = findobj(vctFig, 'Tag', ['editX', num2str(idx)]);
            eY = findobj(vctFig, 'Tag', ['editY', num2str(idx)]);
            
            [files, pth] = uigetfile('*.xlsx', ['子图 ', num2str(idx), ' 导入 (最多选择 8 个 Excel 文件)'], 'MultiSelect', 'on');
            if isequal(files, 0); return; end
            if ~iscell(files); files = {files}; end
            
            % 更新该子图的文件计数
            vctFileCounter(idx) = length(files);
            
            % 计算当前图例的起始 P 编号
            startP = sum(vctFileCounter(1:idx-1)); 
            
            cla(curAx);
            xC = str2double(eX.String); 
            yC_raw = str2double(eY.String); 
            
            % --- 识别全量程 ---
            all_data = cell(1, length(files));
            minX = inf; maxX = -inf; minY = inf; maxY = -inf;
            for i = 1:length(files)
                data = readmatrix(fullfile(pth, files{i}));
                yC = yC_raw; if yC == -1; yC = size(data, 2); end 
                XV = data(:, xC); YV = data(:, yC);
                all_data{i} = {XV, YV};
                minX = min(minX, min(XV)); maxX = max(maxX, max(XV));
                minY = min(minY, min(YV)); maxY = max(maxY, max(YV));
            end
            % --- 修正版：彻底清除内容而不破坏坐标轴对象 ---
            hold(curAx, 'off'); % 暂时关闭 hold
            cla(curAx);         % 清除轴内所有图形对象
            legend(curAx, 'off'); % 移除旧图例
            
            % 恢复基础环境
            grid(curAx, 'on'); hold(curAx, 'on'); box(curAx, 'on');
            title(curAx, titles{idx}, 'FontSize', 12, 'FontWeight', 'bold'); 
            
            if ~isinf(minX) 
                xlim(curAx, [minX*0.95, maxX*1.05]); 
                ylim(curAx, [minY*0.95, maxY*1.05]);
            end
            
                       
            % --- 绘图 (全局 P 编号逻辑) ---
            vctCols = lines(32); % 扩大颜色池
            vctMkrs = {'o', 's', '^', 'd', 'v', 'p'};
            for i = 1:length(files)
                XV = all_data{i}{1}; YV = all_data{i}{2}; [XV_s, sIdx] = sort(XV);
                
                % 计算全局 P 编号
                globalID = startP + i;
                
                % 确保同一幅图颜色不重复，形状随全局 ID 变化
                scatter(curAx, XV, YV, 70, vctCols(globalID,:), vctMkrs{mod(globalID-1, length(vctMkrs))+1}, ...
                        'filled', 'DisplayName', ['P', num2str(globalID)], 'MarkerFaceAlpha', 0.65);
                plot(curAx, XV_s, YV(sIdx), '--', 'Color', [vctCols(globalID,:) 0.3], 'HandleVisibility', 'off');
            end
            legend(curAx, 'show', 'Location', 'northeastoutside', 'FontSize', 7);
            xlabel(curAx, 'Maximum Rubber Particle Size (mm)'); ylabel(curAx, 'Compressive Strength (MPa)');
        end
    end
 end % <--- 保持文件末尾原有的总 end
    
