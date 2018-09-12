from . import views
from django.urls import path

urlpatterns = [
	path('home', views.home, name="home"),
	path('test', views.test, name="test"),
	path('displaydata', views.displaydata, name="displaydata"),
	path('calculate_algo', views.calculate_algo, name="calculate_algo"),
	path('setpara', views.setpara, name="setpara"),
	path('predd', views.predd, name="predd"),
	path('f_result', views.f_result, name="f_result"),
]