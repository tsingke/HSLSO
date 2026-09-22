function plotPath(path, startPos, goalPos, colorSpec, labelName)
    fullPath = [startPos; path; goalPos];

    t  = 1:size(fullPath, 1);
    ts = linspace(1, size(fullPath, 1), 200);

    xs = pchip(t, fullPath(:,1), ts);
    ys = pchip(t, fullPath(:,2), ts);
    zs = pchip(t, fullPath(:,3), ts);

    plot3(xs, ys, zs, colorSpec, 'LineWidth', 2, 'DisplayName', labelName);
    hold on;

end
