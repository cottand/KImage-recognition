@file:Suppress("SpellCheckingInspection")

import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
  kotlin("jvm") version "1.3.60"
}

group = "org.cottand"
version = "1.0-SNAPSHOT"

repositories {
  jcenter()
  mavenCentral()
  maven("https://dl.bintray.com/kotlin/kotlin-numpy")

}

val arrowVersion = "0.10.4"
dependencies {
  implementation(kotlin("stdlib-jdk8"))
  implementation("org.jetbrains:kotlin-numpy:0.1.0")
  implementation("io.arrow-kt:arrow-core:$arrowVersion")
  implementation("io.arrow-kt:arrow-syntax:$arrowVersion")

  testImplementation("org.junit.jupiter:junit-jupiter-api:5.3.1")
  testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.3.1")
}


tasks {
  test {
    useJUnitPlatform()
  }
  compileKotlin {
    kotlinOptions.jvmTarget = "1.8"
  }
  compileTestKotlin {
    kotlinOptions.jvmTarget = "1.8"
  }
}
