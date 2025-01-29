#include "sample_app.hpp"
#include <iostream>

int main()
{
	fprintf(stdout, "Entry point\n");

	SampleAppGUI::GUIApplication application;
	application.window_settings.window_title = "KittlesPT";
	//application.window_settings.initial_window_width = 760;
	application.window_settings.initial_window_width = 960;
	application.window_settings.initial_window_height = 540;

	application.init(std::make_shared<SampleApp::SampleAppWindow>());

	//logging--------
	int gl_ver_minor, gl_ver_major, gl_extensions_num;
	glGetIntegerv(GL_MAJOR_VERSION, &gl_ver_major);
	glGetIntegerv(GL_MINOR_VERSION, &gl_ver_minor);
	glGetIntegerv(GL_NUM_EXTENSIONS, &gl_extensions_num);
	printf("GL version: %d.%d\nVendor: %s\nRenderer: %s\n", gl_ver_major, gl_ver_minor,
		glGetString(GL_VENDOR), glGetString(GL_RENDERER));
	printf("GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("Extensions used: %d\n", gl_extensions_num);

	if (gl_extensions_num >= 0 && false)
	{
		printf("Using extensions:\n");
		for (int i = 0; i < gl_extensions_num; i++) {
			printf("%s,", glGetStringi(GL_EXTENSIONS, i));
		}
	}
	//------logging

	//===========================================

	fprintf(stdout, "starting mainloop...\n");

	application.run();

	fprintf(stdout, "closing app\n");

	application.destroy();
}