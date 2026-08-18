{{define "signature_func" -}}
```mojo
{{if and (.IsStatic) (ne .Name "__init__")}}@staticmethod
{{end -}}
{{/* Mojo 1.0 signatures already include the callable keyword. */ -}}
{{if .Signature}}{{.Signature}}{{else}}{{if .IsDef}}def{{else}}fn{{end}} {{.Name}}{{end}}
```
{{- end}}
