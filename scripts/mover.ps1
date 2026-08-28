Add-Type @"
using System;
using System.Runtime.InteropServices;
public class MouseMover {
    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern void mouse_event(int dwFlags, int dx, int dy, int dwData, int dwExtraInfo);

    public const int MOUSEEVENTF_MOVE = 0x0001;
    public const int MOUSEEVENTF_ABSOLUTE = 0x8000;
}
"@

# Store the start time
$startTime = Get-Date

# Run for 5 hours
$timeLimit = New-TimeSpan -Hours 7

while ((Get-Date) - $startTime -lt $timeLimit) {
    # Move cursor slightly to the right
    [MouseMover]::mouse_event([MouseMover]::MOUSEEVENTF_MOVE, 1, 0, 0, 0)
    Start-Sleep -Seconds 30

    # Move cursor slightly to the left
    [MouseMover]::mouse_event([MouseMover]::MOUSEEVENTF_MOVE, -1, 0, 0, 0)
    Start-Sleep -Seconds 30
}
