from django.urls import path
from . import views
 
app_name = 'api_prod'
urlpatterns = [
    path('', views.ProdView.as_view()),
    path('id/<int:item_id>', views.ProdView.as_view()),
    path('type/<int:item_type_id>', views.ProdView.as_view()),
    path('chat/<int:item_id>/<str:query>', views.query_to_response)
]