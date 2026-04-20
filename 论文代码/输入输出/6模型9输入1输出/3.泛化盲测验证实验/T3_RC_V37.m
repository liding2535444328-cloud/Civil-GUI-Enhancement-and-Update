function T3RC_V38()
% =====================================================================
% Project: Rubberized Concrete AI Design Platform (V38 Final Edition)
% Features: Centered UI Buttons, Auto-save FIG, ± Error Symbol
% =====================================================================
%% --- Module 1: Initialization ---
try
    baseDir = 'Research_Outputs';
    if ~exist(baseDir, 'dir'); mkdir(baseDir); end
    imgDir = fullfile(baseDir, 'Snapshots'); if ~exist(imgDir, 'dir'); mkdir(imgDir); end
    
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
fig = figure('Name', 'Rubberized Concrete AI Design Platform V38', ...
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
btn_w = 0.12; btn_h = 0.04; 
start_x = 0.70; 

hBtn1 = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Pop FIG (SVG)', ...
          'Units', 'normalized', 'Position', [start_x, 0.46, btn_w, btn_h], 'FontWeight', 'bold', ...
          'BackgroundColor', [0.4 0.2 0.6], 'ForegroundColor', 'w', 'Callback', @pop_fig_window, 'FontName', mainFont, 'Tag', 'btn1');

hBtn2 = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Replay (15 Steps)', ...
          'Units', 'normalized', 'Position', [start_x + btn_w + 0.01, 0.46, btn_w, btn_h], 'FontWeight', 'bold', ...
          'BackgroundColor', [0.6 0.3 0.1], 'ForegroundColor', 'w', 'Callback', @replay_recent, 'FontName', mainFont, 'Tag', 'btn2');

hBtn3 = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Export Snapshot', ...
          'Units', 'normalized', 'Position', [start_x, 0.41, btn_w, btn_h], 'FontWeight', 'bold', ...
          'BackgroundColor', [0.2 0.5 0.3], 'ForegroundColor', 'w', 'Callback', @export_gui, 'FontName', mainFont, 'Tag', 'btn3');

hBtn4 = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Export Excel Log', ...
          'Units', 'normalized', 'Position', [start_x + btn_w + 0.01, 0.41, btn_w, btn_h], 'FontWeight', 'bold', ...
          'BackgroundColor', [0.1 0.4 0.6], 'ForegroundColor', 'w', 'Callback', @export_log_excel, 'FontName', mainFont, 'Tag', 'btn4');

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
        % 克隆整个界面到标准 Figure 窗口
        popFig = figure('Name', 'Full UI Export', 'Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
        % 复制主要 UI 组件，包括按钮
        copyobj([axSHAP, ax3D, axHist, pnlRes, hTable, pnlCtrl, ...
                 findobj(fig, 'Tag', 'btn1'), findobj(fig, 'Tag', 'btn2'), ...
                 findobj(fig, 'Tag', 'btn3'), findobj(fig, 'Tag', 'btn4')], popFig);
        
        % 新功能：自动保存 .fig 到文件夹
        ts = datestr(now, 'yyyy-mm-dd_HHMMSS');
        fname = fullfile(CoreData.baseDir, ['Scientific_Figure_', ts, '.fig']);
        saveas(popFig, fname);
        
        msgbox(['Full UI Fig opened and auto-saved to: ', fname], 'Export Success');
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
    end
end