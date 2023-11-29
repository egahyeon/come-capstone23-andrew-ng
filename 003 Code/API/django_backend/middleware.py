from django.http import HttpResponseForbidden

class WhitelistIPMiddleware:
    # 허용할 IP 목록
    WHITELISTED_IPS = ['127.0.0.1', '118.42.71.190', '210.110.250.118'] # 이곳에 허용할 IP들을 추가하세요.
    # 특정 URL에 대한 IP 필터링을 적용하려면 아래 리스트를 수정하세요.
    PROTECTED_PATHS = ['/pig_info/gpu_server/']

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # IP 확인
        ip = self.get_client_ip(request)
        # 현재 요청 URL이 보호된 경로 중 하나인지 확인
        if request.path_info in self.PROTECTED_PATHS and ip not in self.WHITELISTED_IPS:
            return HttpResponseForbidden("You're not allowed here!")
        response = self.get_response(request)
        return response

    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
