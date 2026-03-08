# Counting starts from 1 for both test and train (and valid).
# 1) Folder names: Lavel_1 -> Level1, Lavel_2 -> Level2, Lavel_3 -> Level3, Lavel_4 -> Level4
# 2) Image names: each image is renamed so its name MATCHES the folder it's in
#    (e.g. images in Level1 folder get "Level1" in the filename, etc.)
$root = "c:\Users\Zaidz\Downloads\CRS"
$splits = @("test", "train", "valid")
$folderMap = @{
    "Lavel_1" = "Level1"
    "Lavel_2" = "Level2"
    "Lavel_3" = "Level3"
    "Lavel_4" = "Level4"
}
# For each folder name, which old filename patterns to replace with that folder name
$folderToPatterns = @{
    "Level1"   = @("levle0")
    "Level2"   = @("levle1")
    "Level3"   = @("levle2")
    "Level4"   = @("levle3")
    "Unlabeled" = @("levle0", "levle1", "levle2", "levle3")
}

foreach ($split in $splits) {
    $splitPath = Join-Path $root $split
    if (-not (Test-Path $splitPath)) { continue }
    # 1. Rename folders
    foreach ($old in $folderMap.Keys) {
        $oldPath = Join-Path $splitPath $old
        $newName = $folderMap[$old]
        if (Test-Path $oldPath) {
            Rename-Item -Path $oldPath -NewName $newName -Force
            Write-Host "Renamed folder: $split\$old -> $split\$newName"
        }
    }
    # 2. Rename images so each image name matches its folder name
    $subfolders = Get-ChildItem -Path $splitPath -Directory
    foreach ($sub in $subfolders) {
        $folderName = $sub.Name
        $patterns = $folderToPatterns[$folderName]
        if (-not $patterns) { continue }
        $files = Get-ChildItem -Path $sub.FullName -Filter "*.jpg" -File
        foreach ($f in $files) {
            $base = $f.BaseName
            $newBase = $base
            foreach ($pattern in $patterns) {
                $newBase = $newBase.Replace($pattern, $folderName)
            }
            if ($newBase -ne $base) {
                $newFileName = $newBase + $f.Extension
                Rename-Item -Path $f.FullName -NewName $newFileName -Force
                Write-Host "Renamed: $($f.Name) -> $newFileName"
            }
        }
    }
}
Write-Host "Done."
