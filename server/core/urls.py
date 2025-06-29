# core/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect

def root_redirect(request):
    """Redirect root URL to a default insurance type or landing page."""
    return redirect('insurance:index', insurance_type='health')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', root_redirect, name='root'),  # Handle the root URL
    path('insurance/', include('insurance.urls', namespace='insurance')),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)