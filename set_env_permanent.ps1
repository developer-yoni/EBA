# AWS 자격 증명 영구 설정 (사용자 환경변수)
# 관리자 권한 필요 없음

Write-Host "=" * 80
Write-Host "🔐 AWS 자격 증명 영구 설정"
Write-Host "=" * 80
Write-Host ""

# 자격 증명 입력
Write-Host "1️⃣ AWS Access Key ID"
$accessKey = Read-Host "   입력"

Write-Host ""
Write-Host "2️⃣ AWS Secret Access Key"
$secretKey = Read-Host "   입력"

Write-Host ""
Write-Host "3️⃣ AWS Session Token (선택사항)"
$sessionToken = Read-Host "   입력 (없으면 Enter)"

# 사용자 환경변수에 저장 (영구적)
if ($accessKey) {
    [System.Environment]::SetEnvironmentVariable("AWS_ACCESS_KEY_ID", $accessKey, "User")
    Write-Host "✅ AWS_ACCESS_KEY_ID 설정 완료"
}

if ($secretKey) {
    [System.Environment]::SetEnvironmentVariable("AWS_SECRET_ACCESS_KEY", $secretKey, "User")
    Write-Host "✅ AWS_SECRET_ACCESS_KEY 설정 완료"
}

if ($sessionToken) {
    [System.Environment]::SetEnvironmentVariable("AWS_SESSION_TOKEN", $sessionToken, "User")
    Write-Host "✅ AWS_SESSION_TOKEN 설정 완료"
}

# 리전 설정
[System.Environment]::SetEnvironmentVariable("AWS_REGION", "ap-northeast-2", "User")
Write-Host "✅ AWS_REGION 설정 완료"

Write-Host ""
Write-Host "=" * 80
Write-Host "✅ 환경변수가 영구적으로 설정되었습니다!"
Write-Host "=" * 80
Write-Host ""
Write-Host "⚠️ 주의: 새 PowerShell 창을 열어야 적용됩니다."
Write-Host ""
Write-Host "다음 단계:"
Write-Host "  1. 이 PowerShell 창을 닫기"
Write-Host "  2. 새 PowerShell 창 열기"
Write-Host "  3. 테스트: python test_connection.py"
Write-Host ""
