# Programa básico de un ReAct Agent

Construimos un ReAct Agent básico, el cual tiene que calcular el número de caracteres de una palabra teniendo que elegir y utilizar una herramienta.

Estos son los pasos:

1.- Indicamos la palabra a calcular su número de caracteres 
2.- Se llama n veces al agente hasta que devuelva una respuesta final
3.- En cada llamada elige una herramienta y calcula el resultado de utilizarla
4.- Almacenamos tanto la elección de la herramiento como el resultado
5.- Se vuelve a indicar el agente que con la información anterior puede llegar a una conclusión
6.- Llega a una respuesta final si tiene sentido el razonamiento y termina el proceso. En otro caso se repite el bucle