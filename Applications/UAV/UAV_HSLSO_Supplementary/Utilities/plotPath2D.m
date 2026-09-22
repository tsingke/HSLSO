function plotPath2D(path, startPos, goalPos, colorSpec, labelName)
    fullPath = [startPos; path; goalPos];

    t  = 1:size(fullPath, 1);
    ts = linspace(1, size(fullPath, 1), 200);

    xs = pchip(t, fullPath(:,1), ts);
    ys = pchip(t, fullPath(:,2), ts);

    plot(xs, ys, colorSpec, 'LineWidth', 2, 'DisplayName', labelName);
    hold on;

    scatter(fullPath(:,1), fullPath(:,2), 18, 'filled', 'HandleVisibility', 'off');
end
