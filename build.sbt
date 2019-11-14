name := "project"

version := "0.1"

scalaVersion := "2.11.12"

dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-core" % "2.10.0"
dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-databind" % "2.10.0"
dependencyOverrides += "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.10.0"


libraryDependencies ++= Seq("org.apache.spark" %% "spark-core" % "2.4.3",
                            "org.apache.spark" %% "spark-sql" % "2.4.3",
                            "org.apache.spark" %% "spark-mllib" % "2.4.3",
                            "org.apache.spark" %% "spark-streaming" % "2.4.3",
                            "org.apache.bahir" %% "spark-streaming-twitter" % "2.3.4",
                            "org.slf4j" % "slf4j-log4j12" % "1.7.26"
                        )
