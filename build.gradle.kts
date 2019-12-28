@file:Suppress("SpellCheckingInspection")

plugins {
  kotlin("jvm") version "1.3.61"
  id("org.jmailen.kotlinter") version "2.2.0"
}

kotlinter.indentSize = 2

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
  implementation("org.jetbrains:kotlin-numpy:0.1.1")
  implementation("io.arrow-kt:arrow-core:$arrowVersion")
  implementation("io.arrow-kt:arrow-syntax:$arrowVersion")
  implementation("org.jetbrains.kotlinx:kotlinx-collections-immutable-jvm:0.3")

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
