name := "IBD_Assignment2"

version := "0.1"

scalaVersion := "2.12.8"

dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-core" % "2.10.0"
dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-databind" % "2.10.0"
dependencyOverrides += "com.fasterxml.jackson.module" % "jackson-module-scala_2.12" % "2.10.0"


libraryDependencies ++= Seq("org.apache.spark" %% "spark-core" % "2.4.4",
                            "org.apache.spark" %% "spark-sql" % "2.4.4",
                            "org.apache.spark" %% "spark-mllib" % "2.4.4"
                        )

libraryDependencies += "com.github.scopt" %% "scopt" % "3.7.1"